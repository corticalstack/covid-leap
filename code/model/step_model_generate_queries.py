# See UKPLab for example BEIR benchmarking which this script heavily models from
#https://github.com/UKPLab/beir

import argparse
import os
import re
import pandas as pd
import torch
from azureml.core import Run
from azure.storage.blob import BlobServiceClient
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sqlalchemy import create_engine
import tqdm


class GenerateQueries:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.get_runtime_arguments()

        self.df = pd.DataFrame()
        self.df_gq = pd.DataFrame(columns=['query', 'paragraph'])
        self.dataset = None
        self.corpus_embeddings = None

        print('Cuda:', torch.zeros(1).cuda())
        print('Version cuda:', torch.version.cuda)
        print('Backend cudnn version:', torch.backends.cudnn.version())
        print('Torch version:', torch.__version__)

        if torch.cuda.is_available():
            print('Cuda is available')
        else:
            print('Cuda is unavailable')

        # PSQL db
        self.db_conn_path = None
        self.sqlalchemy_engine = None
        self.psql_host = None
        self.psql_user = None
        self.psql_pwd = None
        self.dbname = 'analytics-shared-repository'

        self.blob_connect_string = None
        self.get_os_environ_variables()

        self.set_db_conn_path()
        self.set_sqlalchemy_engine()

        self.get_corpus_size()
        self.get_corpus_random_sample()

        self.tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
        self.model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')

        self.model.eval()

        print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # Select the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.prepare_corpus()
        self.generate_paragraph_queries()
        self.cleanse_queries()
        self.remove_duplicate_queries()
        self.output_queries_as_blob()

    @staticmethod
    def clean_sentences(sent):
        sent = re.sub(r'http\S+', 'URL', sent, flags=re.MULTILINE)  # Remove URLs
        sent = re.sub(r'[^\w\s]', '', sent)  # Remove punctuation
        if sent:
            return sent

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--sample_size_percent',
            type=int,
            help='Percentage of dataset to sample for egenrating queries'
        )
        parser.add_argument(
            '--queries_per_paragraph',
            type=int,
            help='Number of queries per paragraph to generate'
        )
        parser.add_argument(
            '--max_paragraph_length',
            type=int,
            help='Max length of paragraph'
        )
        parser.add_argument(
            '--max_query_length',
            type=int,
            help='Max query length'
        )
        self.args = parser.parse_args()

        print('Sample size percent: {}'.format(str(self.args.sample_size_percent)))
        print('Queries per paragraph: {}'.format(str(self.args.queries_per_paragraph)))
        print('Max paragraph length: {}'.format(str(self.args.max_paragraph_length)))
        print('Max query length: {}'.format(str(self.args.max_query_length)))

    def get_os_environ_variables(self):
        print('--- Getting OS Environment Variables')
        self.psql_host = os.environ.get('AZ_RP_PSQL_HOST')
        self.psql_user = os.environ.get('AZ_RP_PSQL_USER')
        self.psql_pwd = os.environ.get('AZ_RP_PSQL_PWD')
        self.blob_connect_string = os.environ.get('AZ_RP_MLW_BLOB_CONNECT_STRING')

    def set_db_conn_path(self):
        print('--- Setting Db Connection')
        self.db_conn_path = 'postgresql://' + self.psql_user + ':' + self.psql_pwd + '@' + self.psql_host + ':5432/' + \
                            self.dbname

    def set_sqlalchemy_engine(self):
        print('--- Setting Db Engine')
        self.sqlalchemy_engine = create_engine(self.db_conn_path, connect_args={'sslmode': 'require'})

    def get_corpus_size(self):
        print('--- Get corpus size')
        conn = self.sqlalchemy_engine.connect()
        sql_string = "SELECT count(*) FROM pub_article"
        _df = pd.read_sql(sql_string, conn)
        self.corpus_size = _df.iloc[0]['count']
        print('Corpus contains {} articles'.format(str(self.corpus_size)))

    def get_corpus_random_sample(self):
        print('--- Get corpus random sample')
        conn = self.sqlalchemy_engine.connect()
        sql_string = "SELECT a.hash_id, a.title, " \
                     "t.body_text " \
                     "FROM pub_article a " \
                     "TABLESAMPLE BERNOULLI(" + str(self.args.sample_size_percent) + ') ' \
                     "INNER JOIN pub_body_text t ON t.hash_id = a.hash_id"

        self.df = pd.read_sql(sql_string, conn)
        #samples = int(len(self.df) / 5)
        #self.df = self.df[:samples]
        self.df = self.df[:1000]
        print('Number of PSQL database sample articles: {}'.format(str(len(self.df))))

    def prepare_corpus(self):
        print('--- Prepare Corpus')

        self.df['paragraph'] = self.df['body_text'].str.split('\n', expand=False)
        self.df.drop(columns=['body_text'], inplace=True)

        self.df['paragraph'] = self.df['paragraph'].apply(lambda x: [y for y in x if len(y) > 100])

        self.df = self.df.explode('paragraph')
        self.df = self.df[['hash_id', 'title', 'paragraph']]
        self.df.reset_index(inplace=True)

        self.df.rename(columns={'index': 'title_index'}, inplace=True)
        self.df.reset_index(inplace=True)
        self.df.rename(columns={'index': 'paragraph_index'}, inplace=True)
        self.df['id'] = self.df['hash_id'].astype(str) + '_' + self.df['paragraph_index'].astype(str)

        self.df['paragraph_processed'] = self.df['paragraph']
        self.df = self.df[self.df['paragraph_processed'].notna()]

        print('Preparing cleansed paragraphs')
        self.df['paragraph_processed'] = self.df['paragraph_processed'].apply(lambda x: self.clean_sentences(x))
        self.df = self.df[self.df['paragraph_processed'].notna()]

        print('Number of article paragraphs to be embedded: {}'.format(str(len(self.df))))

        print('Mean length of paragraph processed:', self.df['paragraph_processed'].apply(lambda x: len(x)).mean())
        print('Min length of paragraph processed:', self.df['paragraph_processed'].apply(lambda x: len(x)).min())
        print('Max length of paragraph processed:', self.df['paragraph_processed'].apply(lambda x: len(x)).max())

        print('Mean length of title:', self.df['title'].apply(lambda x: len(x)).mean())
        print('Min length of title:', self.df['title'].apply(lambda x: len(x)).min())
        print('Max length of title:', self.df['title'].apply(lambda x: len(x)).max())

    def generate_paragraph_queries(self):
        print('Generate paragraph queries')

        batch_size = 8
        paragraphs = self.df['paragraph_processed'].to_list()

        for start_idx in tqdm.trange(0, len(paragraphs), batch_size):
            sub_paragraphs = paragraphs[start_idx:start_idx + batch_size]
            inputs = self.tokenizer.prepare_seq2seq_batch(sub_paragraphs, max_length=self.args.max_paragraph_length,
                                                          truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=self.args.max_query_length,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=self.args.queries_per_paragraph)

            for idx, out in enumerate(outputs):
                query = self.tokenizer.decode(out, skip_special_tokens=True)
                para = sub_paragraphs[int(idx / self.args.queries_per_paragraph)]

                query = query.replace("\t", " ").strip()
                para = para.replace("\t", " ").strip()
                self.df_gq = self.df_gq.append(
                    {'query': query,
                     'paragraph': para},
                    ignore_index=True)

        print('Number of query paragraph combinations generated:', str(len(self.df_gq)))

    def cleanse_queries(self):
        print('--- Cleanse queries')
        self.df_gq['query'] = self.df_gq['query'].apply(lambda x: self.clean_sentences(str(x)))
        self.df_gq.replace(r"^ +| +$", r"", regex=True, inplace=True)
        self.df_gq['query'] = self.df_gq['query'].str.replace('\d+', '')

        self.df_gq.fillna('', inplace=True)
        self.df_gq = self.df_gq[self.df_gq['query'].notna()]

    def remove_duplicate_queries(self):
        print('--- Removing duplicate training examples')
        self.df_gq.drop_duplicates(inplace=True)
        print('Size after delete duplicates:', str(len(self.df_gq)))

    def output_queries_as_blob(self):
        print('--- Output generated queries to Azure blob storage')

        print('Number of query paragraph combinations:', str(len(self.df_gq)))
        container_name = 'modelling'
        blob_name = 'cord19_generated_queries_for_model_training.csv'
        blob_service_client = BlobServiceClient.from_connection_string(self.blob_connect_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        output = self.df_gq.to_csv(header=False, index=False, sep='\t', encoding='utf-8')
        blob_client.upload_blob(output, overwrite=True)


if __name__ == '__main__':
    print('--- Model generate queries')
    generate_queries = GenerateQueries()
    print('--- Model generate queries')
