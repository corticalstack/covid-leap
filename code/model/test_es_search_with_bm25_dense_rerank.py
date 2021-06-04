from elasticsearch import Elasticsearch, helpers

import pandas as pd
import numpy as np
import copy
from sentence_transformers import SentenceTransformer, util, models
import time
import os
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from sentence_transformers import SentenceTransformer, InputExample, losses, models, CrossEncoder
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing


class TestEs:
    def __init__(self):
        self.top_n = 5
        self.es_index_name = 'pub_text'
        self.es_host = os.environ.get('AZ_RP_ES_HOST')
        self.es_user = os.environ.get('AZ_RP_ES_USER')
        self.es_pwd = os.environ.get('AZ_RP_ES_PWD')

        self.es_conn = Elasticsearch([self.es_host], http_auth=(self.es_user, self.es_pwd), scheme='https',
                                     port=443, timeout=30)

        self.svc_key = 'xxx'

        self.svc_pr = ServicePrincipalAuthentication(
            tenant_id='xxx',
            service_principal_id='xxx',
            service_principal_password=self.svc_key)

        self.ws = Workspace(
            subscription_id="xxx",
            resource_group="xxx",
            workspace_name="xxx",
            auth=self.svc_pr
        )

        self.df_results = pd.DataFrame(columns=['strategy', 'q_id', 'score', 'question', 'took', 'hash_id', 'title',
                                                'publish_year', 'journal', 'topic_id', 'doi', 'text', 'is_coronavirus',
                                                'is_coronavirus_title', 'is_sars_cov2', 'is_sars_cov2_title',
                                                'is_sars_cov', 'is_sars_cov_title', 'is_mers', 'is_mers_title'])

        self.q_counter = 1

        self.question_embedding = None
        self.neural_model_path = Model.get_model_path('c19gq_ance_msmarco_passage', None, self.ws)
        self.neural_model_embedding = SentenceTransformer(self.neural_model_path)
        if torch.cuda.is_available():
            self.neural_model_embedding = self.neural_model_embedding.to(torch.device('cuda'))

        self.ask_question_ir()

        if len(self.df_results):
            self.df_results.to_csv('Test ES Search results.csv', sep='\t', index=False)

    def ask_question_ir(self):
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

        while True:
            query = input('Please enter a question: ')
            if query == 'x':
                break

            # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-match-query.html
            # https://coralogix.com/blog/42-elasticsearch-query-examples-hands-on-tutorial/

            print('\nBM25')
            bm25 = self.es_conn.search(index=self.es_index_name, body=self.get_bm25_body(query), size=self.top_n, request_timeout=30)
            for hit in bm25['hits']['hits']:
                self.log_results('BM25', query, bm25['took']/1000, hit)

            print('\nSemantic')
            s_time = time.time()
            self.question_embedding = self.neural_model_embedding.encode(query)
            e_time = time.time()
            q_encoding_took = e_time - s_time

            s_time = time.time()
            sem_search = self.es_conn.search(index=self.es_index_name, body=self.get_semantic_body(), size=self.top_n, request_timeout=50)
            e_time = time.time()

            for hit in sem_search['hits']['hits']:
                self.log_results('Semantic', query, (e_time - s_time) + q_encoding_took, hit)

            hits = sem_search['hits']['hits']
            cross_inp = [[query, hit['fields']['title'][0]] for hit in hits]
            cross_scores = cross_encoder.predict(cross_inp)
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]


            # Discussion on normalising BM25
            # https://stats.stackexchange.com/questions/171589/normalised-score-for-bm25
            #total_bm25_score = df_results['score'].sum()
            #df_results['score'] = df_results['score'] / total_bm25_score
            #print(df_results)

            print('\nBM25 + Neural Reranker results')
            s_time = time.time()
            df_bm25_reranked = self.df_results[
                (self.df_results['strategy'] == 'BM25') & (self.df_results['q_id'] == self.q_counter)].copy()
            df_bm25_reranked['q_vector'] = df_bm25_reranked.apply(lambda x: self.question_embedding, axis=1)
            df_bm25_reranked['text_vector'] = df_bm25_reranked.apply(lambda row: self.neural_model_embedding.encode(
                row[5] + row[10]), axis=1)
            q_vector = np.array(list(df_bm25_reranked['q_vector']))
            text_vector = np.array(list(df_bm25_reranked['text_vector']))
            df_bm25_reranked['score'] = np.diag(cosine_similarity(q_vector, text_vector))
            e_time = time.time()
            df_bm25_reranked['took'] = df_bm25_reranked['took'] + (e_time - s_time)
            df_bm25_reranked.sort_values(by=['score'], ascending=False, inplace=True)
            df_bm25_reranked['strategy'] = 'BM25 + SEM'

            df_bm25_reranked.drop(columns=['q_vector', 'text_vector'], inplace=True, axis=1)
            self.df_results = self.df_results.append(df_bm25_reranked, ignore_index=True)

            self.q_counter += 1
            print("\n\n========\n")

    def log_results(self, strategy, q, took, hit):
        result = {'strategy': strategy,
                  'q_id': self.q_counter,
                  'score': hit['_score'],
                  'question': q,
                  'took': took,
                  'hash_id': hit['fields']['hash_id'][0],
                  'title': hit['fields']['title'][0],
                  'publish_year': 0,
                  'journal': '',
                  'topic_id': hit['fields']['topic_id'][0],
                  'doi': '',
                  'pubmed_id': '',
                  'text': hit['fields']['text'][0],
                  'is_coronavirus': hit['fields']['is_coronavirus'][0],
                  'is_coronavirus_title': hit['fields']['is_coronavirus_title'][0],
                  'is_sars_cov2': hit['fields']['is_sars_cov2'][0],
                  'is_sars_cov2_title': hit['fields']['is_sars_cov2_title'][0],
                  'is_sars_cov': hit['fields']['is_sars_cov'][0],
                  'is_sars_cov_title': hit['fields']['is_sars_cov_title'][0],
                  'is_mers': hit['fields']['is_mers'][0],
                  'is_mers_title': hit['fields']['is_mers_title'][0]}

        if 'publish_year' in hit['fields']:
            result['publish_year'] = hit['fields']['publish_year'][0]

        if 'doi' in hit['fields']:
            result['doi'] = hit['fields']['doi'][0]

        if 'pubmed_id' in hit['fields']:
            result['pubmed_id'] = hit['fields']['pubmed_id'][0]

        self.df_results = self.df_results.append(result, ignore_index=True)

    def get_bm25_body(self, q):
        # Note filter restriction to sars-cov-2
        bm25_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": q,
                                "fields": [
                                    "title",
                                    "text"
                                ]
                            }
                        },
                        {
                            "terms": {
                                "is_sars_cov2": [
                                    "true"
                                ]
                            }
                        }
                    ]
                }
            },
            "fields": [
                "hash_id",
                "title",
                "publish_year",
                "journal",
                "topic_id",
                "doi",
                "pubmed_id",
                "text",
                "is_coronavirus",
                "is_coronavirus_title",
                "is_sars_cov2",
                "is_sars_cov2_title",
                "is_sars_cov",
                "is_sars_cov_title",
                "is_mers",
                "is_mers_title"
            ],
            "_source": False
        }
        return bm25_body

    def get_semantic_body(self):
        body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "script_score": {
                                "query": {
                                    "match_all": {}
                                },
                                "script": {
                                    "source": "cosineSimilarity(params.queryVector, doc['text_processed_vector']) + 1.0",
                                    "params": {
                                        "queryVector": self.question_embedding
                                    }
                                }
                            }
                        },
                        {
                            "terms": {
                                "is_sars_cov2": [
                                    "true"
                                ]
                            }
                        }
                    ]
                }
            },
            "fields": [
                "hash_id",
                "title",
                "publish_year",
                "journal",
                "topic_id",
                "doi",
                "pubmed_id",
                "text",
                "is_coronavirus",
                "is_coronavirus_title",
                "is_sars_cov2",
                "is_sars_cov2_title",
                "is_sars_cov",
                "is_sars_cov_title",
                "is_mers",
                "is_mers_title"
            ],
            "_source": False
        }
        return body


if __name__ == '__main__':
    print('--- Test ES Search')
    test_es = TestEs()
    print('--- Test ES completed')



