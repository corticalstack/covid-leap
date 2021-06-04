import argparse
import os
import re
import pandas as pd
import torch
from azureml.core import Run
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from sentence_transformers import SentenceTransformer
import joblib
from sqlalchemy import create_engine


class EmbedCorpus:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.get_runtime_arguments()

        self.df = pd.DataFrame()
        self.dataset = None
        self.corpus_embeddings = None

        self.build_model = True
        self.path_corpus_as_df_chunk = 'corpus_as_df_chunk_'

        if torch.cuda.is_available():
            print('Cuda is available')
        else:
            print('Cuda is unavailable')

        # Model
        self.neural_model_embedding = None
        self.neural_model_path = None
        self.model = None
        self.corpus_embeddings = None

        self.svc_pr = self.set_service_principal()
        self.ws = self.set_workspace(self.svc_pr)

        self.get_registered_neural_model()
        self.set_neural_model_embedding(self.neural_model_path)

        # PSQL db
        self.db_conn_path = None
        self.sqlalchemy_engine = None
        self.psql_host = None
        self.psql_user = None
        self.psql_pwd = None
        self.dbname = 'analytics-shared-repository'

        self.get_os_environ_variables()
        self.set_db_conn_path()
        self.set_sqlalchemy_engine()

        self.corpus_size = 0
        self.corpus_chunk_size = 0
        self.set_chunk_size()
        self.get_corpus_size()

        self.df_chunk_pkl = []
        for idx, offset in enumerate(range(0, self.corpus_size, self.corpus_chunk_size)):
            print('Processing chunk {}'.format(idx))
            self.get_corpus(offset)
            self.prepare_corpus()
            self.embed_corpus()
            self.save_corpus_as_df(idx)

    @staticmethod
    def set_workspace(svc_pr):
        ws = Workspace(
            subscription_id="19518d47-0c8b-4829-a602-c5ced78deb3f",
            resource_group="aci-eur-frc-aa-ss-rg",
            workspace_name="aci-eur-frc-aa-ss-mlw",
            auth=svc_pr)
        return ws

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
            '--dataset_size_percent',
            type=int,
            help='Percentage of dataset to process'
        )
        parser.add_argument(
            '--wrkdir',
            type=str,
            help='Model working directory'
        )
        parser.add_argument(
            '--neural_model_name',
            type=str,
            help='Neural model name'
        )

        self.args = parser.parse_args()

        print('Dataset size percent: {}'.format(str(self.args.dataset_size_percent)))
        print('Working Directory: {}'.format(self.args.wrkdir))
        print('Neural model name: {}'.format(self.args.neural_model_name))

    def set_service_principal(self):
        #svc_key = self.run.get_secret(name='aci-eur-frc-aa-ss-mlw-svc-principal-key')
        svc_key = 'xxx'

        svc_pr = ServicePrincipalAuthentication(
            tenant_id='xxx',
            service_principal_id='xxx',
            service_principal_password=svc_key)

        return svc_pr

    def get_registered_neural_model(self):
        print('--- Get registered neural model')
        #ws = Workspace.from_config()
        self.neural_model_path = Model.get_model_path(self.args.neural_model_name, None, self.ws)

    def set_neural_model_embedding(self, model_path):
        print('--- Set neural model embedding')
        self.neural_model_embedding = SentenceTransformer(self.neural_model_path)
        if torch.cuda.is_available():
            self.neural_model_embedding = self.neural_model_embedding.to(torch.device('cuda'))
            print(self.neural_model_embedding.device)

    def get_os_environ_variables(self):
        print('--- Getting OS Environment Variables')
        self.psql_host = os.environ.get('AZ_RP_PSQL_HOST')
        self.psql_user = os.environ.get('AZ_RP_PSQL_USER')
        self.psql_pwd = os.environ.get('AZ_RP_PSQL_PWD')

    def set_db_conn_path(self):
        print('--- Setting Db Connection')
        self.db_conn_path = 'postgresql://' + self.psql_user + ':' + self.psql_pwd + '@' + self.psql_host + ':5432/' + \
                            self.dbname

    def set_sqlalchemy_engine(self):
        print('--- Setting Db Engine')
        self.sqlalchemy_engine = create_engine(self.db_conn_path, connect_args={'sslmode': 'require'})

    def set_chunk_size(self):
        self.corpus_chunk_size = int((5000 / 100) * self.args.dataset_size_percent)
        print('Chunk size: {}'.format(str(self.corpus_chunk_size)))

    def get_corpus_size(self):
        print('--- Get corpus size')
        conn = self.sqlalchemy_engine.connect()
        sql_string = "SELECT count(*) FROM pub_article"
        _df = pd.read_sql(sql_string, conn)
        self.corpus_size = _df.iloc[0]['count']

        self.corpus_size = int((self.corpus_size / 100) * self.args.dataset_size_percent)
        print('Number of corpus articles to be processed: {}'.format(self.corpus_size))

    def get_corpus(self, offset):
        print('--- Get Corpus')
        print('SQL offset is {}'.format(str(offset)))

        conn = self.sqlalchemy_engine.connect()
        sql_string = "SELECT a.hash_id, a.title, a.doi, a.pubmed_id, a.publish_year, a.journal, a.url, " \
                     "a.is_coronavirus, a.is_coronavirus_title, a.is_sars_cov2, a.is_sars_cov2_title, " \
                     "a.is_sars_cov, a.is_sars_cov_title, a.is_mers, a.is_mers_title, a.source, " \
                     "t.body_text, " \
                     "m.author_count, m.paper_citation_count, m.paper_pagerank, m.score_mf1, m.score_mf2, " \
                     "m.score_mf3, m.score_mf4, " \
                     "c.topic_id " \
                     "FROM pub_article a " \
                     "INNER JOIN pub_body_text t ON t.hash_id = a.hash_id " \
                     "INNER JOIN pub_paper_metrics m ON m.hash_id = a.hash_id " \
                     "INNER JOIN pub_document_topic c ON c.hash_id = a.hash_id " \
                     "order by a.hash_id " \
                     "limit " + str(self.corpus_chunk_size) + " offset " + str(offset)

        self.df = pd.read_sql(sql_string, conn)
        print('Number of articles read from PSQL database: {}'.format(str(len(self.df))))

    def prepare_corpus(self):
        print('--- Prepare Corpus')

        # Prepare paragraph
        self.df['paragraph'] = self.df['body_text'].str.split('\n', expand=False)
        self.df.drop(columns=['body_text'], inplace=True)

        self.df['paragraph'] = self.df['paragraph'].apply(lambda x: [y for y in x if len(y) > 100])

        self.df = self.df.explode('paragraph')
        self.df = self.df[['hash_id', 'title', 'paragraph', 'doi', 'pubmed_id', 'publish_year', 'journal', 'url',
                           'is_coronavirus', 'is_coronavirus_title', 'is_sars_cov2', 'is_sars_cov2_title',
                           'is_sars_cov', 'is_sars_cov_title', 'is_mers', 'is_mers_title', 'source', 'author_count',
                           'paper_citation_count', 'paper_pagerank', 'score_mf1', 'score_mf2', 'score_mf3',
                           'score_mf4', 'topic_id']]
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

        # Prepare title
        self.df['title_processed'] = self.df['title']
        self.df['title_processed'] = self.df['title_processed'].apply(lambda x: self.clean_sentences(x))
        self.df = self.df[self.df['title_processed'].notna()]

        print('Number of article paragraphs to be embedded: {}'.format(str(len(self.df))))

    def embed_corpus(self):
        print('Embed corpus title and paragraphs as sentence embeddings')
        self.corpus_embeddings = self.neural_model_embedding.encode(
            (self.df['title_processed'] + ' ' + self.df['paragraph_processed']).tolist(), show_progress_bar=True)
        self.df['paragraph_model_embedding'] = self.corpus_embeddings.tolist()

    def save_corpus_as_df(self, idx):
        print('Save corpus as dataframe')
        _path = self.args.wrkdir + '/' + self.path_corpus_as_df_chunk + str(idx) + '.pkl'
        os.makedirs(self.args.wrkdir, exist_ok=True)
        joblib.dump(self.df, _path)


if __name__ == '__main__':
    print('--- Embed corpus started')
    embed_corpus = EmbedCorpus()
    print('--- Embed corpus completed')
