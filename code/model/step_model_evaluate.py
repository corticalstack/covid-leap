import os
import argparse
import operator
import pandas as pd
import numpy as np
from azureml.core import Run
from azureml.core.model import Model
from azureml.core import Workspace
from azure.storage.blob import BlobServiceClient
from azureml.core.authentication import ServicePrincipalAuthentication
from sentence_transformers import SentenceTransformer, InputExample, losses, models, CrossEncoder
import torch
from elasticsearch import Elasticsearch, helpers

import logging


from beir import util, LoggingHandler
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking import Rerank

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

import random
import requests

import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict


class CustomModel:
    def __init__(self, model_path=None, **kwargs):
        self.model = SentenceTransformer(model_path)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        return np.asarray(self.model.encode(queries, batch_size=batch_size, **kwargs))

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        sentences = [(doc['title'] + "  " + doc['text']).strip() for doc in corpus]
        return np.asarray(self.model.encode(sentences, batch_size=batch_size, **kwargs))


class ModelEvaluate:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.get_runtime_arguments()

        self.blob_connect_string = None
        self.get_os_environ_variables()

        if torch.cuda.is_available():
            print('Cuda is available')
        else:
            print('Cuda is unavailable')

        torch.cuda.empty_cache()

        self.working_dir = 'temp'
        self.working_dir_full_path = None
        self.set_working_dir()

        self.ws = self.set_workspace()

        self.train_examples = []
        self.df = pd.DataFrame()
        self.df_bm_covid_qa = pd.DataFrame()

        self.qrels = None
        self.corpus = None
        self.queries = None
        self.retriever = None
        self.results = None
        self.bm25_results = None
        self.df_performance = pd.DataFrame(columns=['model', 'bm', 'metric', 'score'])

        self.model = None
        self.model_path = None
        self.neural_model_embedding = None
        self.neural_model_path = None

        self.cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.bm_suite()
        #self.output_dataset()
        self.output_eval_as_blob()

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--benchmark_set',
            nargs='+',
            help='Benchmark set'
        )
        parser.add_argument(
            '--base_model_set',
            nargs='+',
            default=[],
            required=False,
            help='Base model set'
        )
        parser.add_argument(
            '--gq_model_set',
            nargs='+',
            default=[],
            required=False,
            help='Generated query trained model set'
        )
        parser.add_argument(
            '--ce_model_set',
            nargs='+',
            default=[],
            required=False,
            help='Cross-encoder model set'
        )

        self.args = parser.parse_args()
        print('Benchmark set: {}'.format(self.args.benchmark_set))
        print('Base model set: {}'.format(self.args.base_model_set))
        print('Generated query trained model set: {}'.format(self.args.gq_model_set))
        print('Cross-encoder model set: {}'.format(self.args.ce_model_set))

    def get_os_environ_variables(self):
        print('--- Getting OS Environment Variables')
        self.blob_connect_string = os.environ.get('AZ_RP_MLW_BLOB_CONNECT_STRING')

    def set_working_dir(self):
        print('--- Setting working directory')
        self.working_dir_full_path = os.path.join(os.getcwd(), self.working_dir)
        print('Work directory:', self.working_dir_full_path)

    def set_workspace(self):
        print('--- Set workspace')
        svc_key = 'xxx'

        svc_pr = ServicePrincipalAuthentication(
            tenant_id='xxx',
            service_principal_id='xxx',
            service_principal_password=svc_key)

        ws = Workspace(
            subscription_id="xxx",
            resource_group="xxx",
            workspace_name="xxx",
            auth=svc_pr
        )

        return ws

    def get_registered_neural_model(self, model):
        print('--- Get registered neural model')
        self.neural_model_path = Model.get_model_path(model, None, self.ws)

    def set_neural_model_embedding(self):
        print('--- Set neural model embedding')
        self.neural_model_embedding = SentenceTransformer(self.neural_model_path)

    def neural_model_encode_text(self, text):
        return self.neural_model_embedding.encode(text)

    def bm_suite(self):
        print('--- Benchmark Suite')

        # List of benchmarks
        # https://github.com/UKPLab/beir


        model_set = []
        #if self.args.base_model_set is not None:
        #    model_set.extend(self.args.base_model_set)

        if self.args.gq_model_set is not None:
            model_set.extend(self.args.gq_model_set)

        #if self.args.ce_model_set is not None:
        #    model_set.extend(self.args.ce_model_set)

        #model_set = self.args.base_model_set + self.args.gq_model_set

        es_hostname = 'https://elastic:GQY7H5JXJGa0dTAlytYGP61V@2ef6f411bbcc465b947e48b48db1518c.francecentral.azure.elastic-cloud.com:9243'
        es_index_init = True  # True, will delete existing index with same name and reindex all documents

        # Evaluate models with each benchmark
        for bm in self.args.benchmark_set:
            dataset_path = None
            print('\n\nEvaluating with benchmark {}'.format(bm))

            if bm == 'covid-qa':
                dataset_path = self.bm_download_covid_qa()
            else:
                dataset_path = self.bm_download_beir(bm)
                self.corpus, self.queries, self.qrels = GenericDataLoader(dataset_path).load(split='test')

            for m in model_set:
                print('\n', m)

                if m in self.args.gq_model_set:
                    self.get_registered_neural_model(m)
                    self.set_neural_model_embedding()

                if bm == 'covid-qa':
                    if m != 'BM25':
                        self.bm_covid_qa(dataset_path, m)
                        self.plot_text_length(bm, m)
                    continue

                if m == 'BM25':
                    model = BM25(index_name=bm, hostname=es_hostname, initialize=es_index_init)
                elif m in self.args.gq_model_set:
                    model = DRES(CustomModel(self.neural_model_path), batch_size=16)
                else:
                    model = DRES(models.SentenceBERT(m), batch_size=16)
                self.retriever = EvaluateRetrieval(model, score_function='cos_sim')
                self.results = self.retriever.retrieve(self.corpus, self.queries)
                ndcg, _map, recall, precision = self.retriever.evaluate(self.qrels, self.results,
                                                                        self.retriever.k_values)
                self.log_score(m, bm, 'ndcg10', ndcg['NDCG@10'])
                self.print_top_k_docs()
                self.plot_text_length(bm, m)

                if m == 'BM25':
                    self.bm25_results = self.results  # Persist BM25 results for reranker

                if m in self.args.gq_model_set and self.bm25_results:
                    reranker_strategy = 'BM25 ' + m + ' Reranker'
                    print('Reranking with ', reranker_strategy)
                    self.results = self.retriever.rerank(self.corpus, self.queries, self.bm25_results, top_k=100)
                    ndcg, _map, recall, precision = self.retriever.evaluate(self.qrels, self.results,
                                                                            self.retriever.k_values)
                    self.log_score(reranker_strategy, bm, 'ndcg10', ndcg['NDCG@10'])
                    self.print_top_k_docs()
                    self.plot_text_length(bm, reranker_strategy)

                if m == 'c19gq_ance_msmarco_passage':
                    reranker_strategy = m + ' Cross-Encoder Reranker'
                    print('Reranking with ', reranker_strategy)

                    reranker = Rerank(self.cross_encoder_model, batch_size=128)
                    rerank_results = reranker.rerank(self.corpus, self.queries, self.results, top_k=100)
                    ndcg, _map, recall, precision = self.retriever.evaluate(self.qrels, rerank_results,
                                                                            self.retriever.k_values)
                    self.log_score(reranker_strategy, bm, 'ndcg10', ndcg['NDCG@10'])
                    self.print_top_k_docs()
                    self.plot_text_length(bm, reranker_strategy)

        print('Finished evaluation')

    def print_top_k_docs(self):
        top_n_docs = 10
        query_id, scores = random.choice(list(self.results.items()))
        scores_sorted = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print('\nQuery: {}'.format(self.queries[query_id]))
        for rank in range(top_n_docs):
            doc_id = scores_sorted[rank][0]
            print('Rank {}:  ID: {}  Title: {}  Text: {}'.format(str(rank + 1),
                                                                 doc_id,
                                                                 self.corpus[doc_id].get('title'),
                                                                 self.corpus[doc_id].get('title')))

    def plot_text_length(self, bm, m):
        df_temp = pd.DataFrame(columns=['length', 'metric'])

        if bm == 'covid-qa':
            top_n = 20  # 488 questions
            fig_name = bm + ' ' + m + ' - Combined Question Length Top ' + str(top_n)
            self.df_bm_covid_qa.sort_values(by=['pred'], ascending=False, inplace=True)
            for index, row in self.df_bm_covid_qa[:top_n].iterrows():
                q_length = float(len(row[0]) + len(row[0]))
                df_temp = df_temp.append({'length': q_length, 'metric': row[5]}, ignore_index=True)
            self.plot_violin(fig_name, 'Combined Question Length', df_temp)
        else:
            top_n = 10
            max_doc_length = 2000
            fig_name = bm + ' ' + m + ' - Passage Length Top ' + str(top_n)
            query_results = list(self.results.items())
            for qr in query_results:
                sorted_qr_d = dict(sorted(qr[1].items(), key=operator.itemgetter(1), reverse=True))
                qr_top_n = list(sorted_qr_d.items())[:top_n]
                for r in qr_top_n:
                    doc_length = float(len(self.corpus[r[0]]['text']))
                    if 0 < doc_length < max_doc_length:
                        df_temp = df_temp.append({'length': doc_length, 'metric': r[1]}, ignore_index=True)
            self.plot_violin(fig_name, 'Passage Length', df_temp)

    def plot_violin(self, fig_name, xlabel, data):
        fig, ax = plt.subplots(figsize=(20, 14))
        plt.xlabel(xlabel, fontsize=24)
        sns.violinplot(data=data, x='length', inner='quartile', orient='h')
        ax.tick_params(axis='both', which='major', labelsize=20)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close()

    def bm_download_covid_qa(self):
        os.makedirs(self.working_dir_full_path, exist_ok=True)
        dataset_path = self.working_dir_full_path + '/eval_question_similarity_en.csv'
        chunk_size: int = 128
        url = 'https://raw.githubusercontent.com/deepset-ai/COVID-QA/master/data/eval_question_similarity_en.csv'

        r = requests.get(url, stream=True)
        with open(dataset_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

        return dataset_path

    def bm_download_beir(self, bm):
        url = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip'.format(bm)
        os.makedirs(self.working_dir, exist_ok=True)
        return util.download_and_unzip(url, self.working_dir)

    def bm_covid_qa(self, dataset_path, model_name):
        model = None
        if model_name in self.args.gq_model_set:
            model = SentenceTransformer(self.neural_model_path)
        elif model_name in self.args.base_model_set:
            model = SentenceTransformer(model_name)
        elif model_name in self.args.ce_model_set:
            model = CrossEncoder('cross-encoder/' + model_name)

        if model is None:
            return

        self.df_bm_covid_qa = pd.read_csv(dataset_path)
        if type(model).__name__ == 'CrossEncoder':
            q1 = self.df_bm_covid_qa['question_1'].to_list()
            q2 = self.df_bm_covid_qa['question_2'].to_list()
            new_lst = [list(x) for x in zip(q1, q2)]
            cross_scores = model.predict(new_lst)
            cross_scores = minmax_scale(cross_scores, feature_range=(0, 1), axis=0, copy=True)
            self.df_bm_covid_qa['pred'] = cross_scores.round(0)
        else:
            self.df_bm_covid_qa['q1_vector'] = self.df_bm_covid_qa.apply(lambda row: model.encode(row[0]), axis=1)
            self.df_bm_covid_qa['q2_vector'] = self.df_bm_covid_qa.apply(lambda row: model.encode(row[1]), axis=1)
            q1_vector = np.array(list(self.df_bm_covid_qa['q1_vector']))
            q2_vector = np.array(list(self.df_bm_covid_qa['q2_vector']))
            self.df_bm_covid_qa['pred'] = np.diag(cosine_similarity(q1_vector, q2_vector))

        y_true = self.df_bm_covid_qa['similar'].values
        y_pred = self.df_bm_covid_qa['pred'].values
        mean_diff = np.mean(np.abs(y_true - y_pred))
        roc_auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred.round(0))

        self.log_score(model_name, 'covid-qa', 'ROC_AUC', roc_auc)
        self.log_score(model_name, 'covid-qa', 'F1', f1)

    def log_score(self, model_name, bm, metric, score):
        self.df_performance = self.df_performance.append({'model': model_name, 'bm': bm, 'metric': metric,
                                                          'score': score}, ignore_index=True)

    def output_dataset(self):
        print('--- Output Dataset')
        perf_results_path = self.working_dir_full_path + '/retriever_model_eval_results.csv'
        self.df_performance.to_csv(perf_results_path, index=False)
        print(self.df_performance)
        print('Output performance results: {}'.format(perf_results_path))

    def output_eval_as_blob(self):
        print('--- Output evaluation results to Azure blob storage')

        container_name = 'modelling'
        blob_name = 'retriever_model_eval_results.csv'
        blob_service_client = BlobServiceClient.from_connection_string(self.blob_connect_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        output = self.df_performance.to_csv(header=True, encoding='utf-8')
        blob_client.upload_blob(output, overwrite=True)

    def offline_save_fig(self, name):
        if 'OfflineRun' in self.run._identity:
            full_path = 'plots/' + name + '.png'
            plt.savefig(full_path, dpi=300, format='png')


if __name__ == '__main__':
    print('--- Model evaluate started')
    model_evaluate = ModelEvaluate()
    print('--- Model evaluate  completed')


