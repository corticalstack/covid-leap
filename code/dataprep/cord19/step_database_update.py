import argparse
import os
import pandas as pd
import numpy as np
from azureml.core import Run
from sqlalchemy import create_engine, exc


class DatabaseUpdate:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.get_runtime_arguments()

        self.db_conn_path = None
        self.sqlalchemy_engine = None

        self.psql_host = None
        self.psql_user = None
        self.psql_pwd = None
        self.dbname = 'analytics-shared-repository'
        self.get_os_environ_variables()

        self.set_db_conn_path()
        self.set_sqlalchemy_engine()

        self.collect_metrics_pre()

        self.df_ref_hash_id = pd.DataFrame()

        self.update_db_from_author_article_metrics()
        self.update_db_from_feature_engineering()
        self.update_db_from_topic_modelling_topics()
        self.update_db_from_topic_modelling_document_topic()
        self.update_db_from_author_info()
        self.update_db_from_authors_by_paper()

        self.update_db_from_trials()

        self.log_metrics_post()

    @staticmethod
    def set_db_update_mode(i):
        if i == 0:
            return 'replace'
        return 'append'

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--input_topic_modelling_topics',
            type=str,
            help='Input topic modelling topics'
        )
        parser.add_argument(
            '--input_topic_modelling_document_topic',
            type=str,
            help='Input topic modelling document topic'
        )
        parser.add_argument(
            '--input_feature_engineering',
            type=str,
            help='Input feature engineering'
        )
        parser.add_argument(
            '--input_author_info',
            type=str,
            help='Input author info'
        )
        parser.add_argument(
            '--input_authors_by_paper',
            type=str,
            help='Input authors by paper'
        )
        parser.add_argument(
            '--input_author_article_metrics',
            type=str,
            help='Input author article metrics'
        )
        parser.add_argument(
            '--input_trials',
            type=str,
            help='Input trials'
        )

        self.args = parser.parse_args()

        print('Input Topic Modelling Topics: {}'.format(self.args.input_topic_modelling_topics))
        print('Input Topic Modelling Document Topic: {}'.format(self.args.input_topic_modelling_document_topic))
        print('Input Feature Engineering: {}'.format(self.args.input_feature_engineering))
        print('Input Author Info: {}'.format(self.args.input_author_info))
        print('Input Authors By Paper: {}'.format(self.args.input_authors_by_paper))
        print('Input Author Article Metrics: {}'.format(self.args.input_author_article_metrics))
        print('Input Trials: {}'.format(self.args.input_trials))

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

    def update_db_from_feature_engineering(self):
        print('--- Update Db From Feature Engineering')
        n_chunk_size = 10000  # chunk row size

        path = self.args.input_feature_engineering + '/processed.csv'
        df = pd.read_csv(path, dtype={
            'paper_id': str,
            'body_text': str,
            'results': str,
            'bibliography': str,
            'subset_source': str,
            'cord_uid': str,
            'sha': str,
            'source': str,
            'title': str,
            'doi': str,
            'pubmed_id': str,
            'abstract': str,
            'publish_time': str,
            'authors': str,
            'journal': str,
            'url': str,
            'hash_id': str,
            'publish_year': int,
            'is_coronavirus': str,
            'is_coronavirus_title': str,
            'is_sars_cov2': str,
            'is_sars_cov2_title': str,
            'is_sars_cov': str,
            'is_sars_cov_title': str,
            'is_mers': str,
            'is_mers_title': str,
        })

        df = self.df_ref_hash_id.merge(df, left_on='hash_id', right_on='hash_id')

        print('Raw Input Specifications')
        print(df.head())
        print(df.columns)
        print(df.dtypes)
        print(df.shape)

        table_name = 'pub_article'
        self.drop_table(table_name)

        df_temp = df[['hash_id', 'title', 'doi', 'pubmed_id', 'publish_time', 'publish_year', 'journal', 'url',
                      'is_coronavirus', 'is_coronavirus_title', 'is_sars_cov2', 'is_sars_cov2_title', 'is_sars_cov',
                      'is_sars_cov_title', 'is_mers', 'is_mers_title']]
        df_temp['publish_time'] = pd.to_datetime(df_temp['publish_time'], errors='ignore').dt.date
        df_temp['source'] = 'cord19'

        df_chunked = [df_temp[i:i + n_chunk_size] for i in range(0, df_temp.shape[0], n_chunk_size)]
        for idx, c in enumerate(df_chunked):
            mode = self.set_db_update_mode(idx)
            self.update_db(c, table_name, mode)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'hash_id')
            conn.execute(sql_string)

        print('Updated table {}'.format(table_name))

        table_name = 'pub_body_text'
        self.drop_table(table_name)

        df_temp = df[['hash_id', 'body_text']]

        df_chunked = [df_temp[i:i + n_chunk_size] for i in range(0, df_temp.shape[0], n_chunk_size)]
        for idx, c in enumerate(df_chunked):
            mode = self.set_db_update_mode(idx)
            self.update_db(c, table_name, mode)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'hash_id')
            conn.execute(sql_string)

        print('Updated table {}'.format(table_name))

        table_name = 'pub_abstract'
        self.drop_table(table_name)

        df_temp = df[['hash_id', 'abstract']]
        df_chunked = [df_temp[i:i + n_chunk_size] for i in range(0, df_temp.shape[0], n_chunk_size)]
        for idx, c in enumerate(df_chunked):
            mode = self.set_db_update_mode(idx)
            self.update_db(c, table_name, mode)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'hash_id')
            conn.execute(sql_string)

        print('Updated table {}'.format(table_name))

    def update_db_from_author_info(self):
        print('--- Update Db From Author Info')
        n_chunk_size = 10000  # chunk row size

        path = self.args.input_author_info + '/processed.csv'
        df = pd.read_csv(path, float_precision='round_trip', dtype={
            'hash_author': str,
            'author': str,
            'paper_count': int,
            'author_citation_count': int,
            'pagerank': np.float64})

        print('Raw Input Specifications')
        print(df.head())
        print(df.columns)
        print(df.dtypes)
        print(df.shape)

        table_name = 'pub_author_metrics'
        self.drop_table(table_name)

        df_chunked = [df[i:i + n_chunk_size] for i in range(0, df.shape[0], n_chunk_size)]
        for idx, c in enumerate(df_chunked):
            mode = self.set_db_update_mode(idx)
            self.update_db(c, table_name, mode)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'hash_author')
            conn.execute(sql_string)

        print('Updated table {}'.format(table_name))

    def update_db_from_topic_modelling_topics(self):
        print('--- Update Db From Topic Modelling Topics')
        n_chunk_size = 10000  # chunk row size

        path = self.args.input_topic_modelling_topics + '/processed.csv'
        df = pd.read_csv(path, dtype={
            'topic_id': int,
            'word': str,
            'distribution': np.float32})

        print('Raw Input Specifications')
        print(df.head())
        print(df.columns)
        print(df.dtypes)
        print(df.shape)

        table_name = 'pub_topics'
        self.drop_table(table_name)

        df_chunked = [df[i:i + n_chunk_size] for i in range(0, df.shape[0], n_chunk_size)]
        for idx, c in enumerate(df_chunked):
            mode = self.set_db_update_mode(idx)
            self.update_db(c, table_name, mode)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'topic_id, word, distribution')
            conn.execute(sql_string)

        print('Updated table {}'.format(table_name))

    def update_db_from_topic_modelling_document_topic(self):
        print('--- Update Db From Topic Modelling Document Topic')
        n_chunk_size = 10000  # chunk row size

        path = self.args.input_topic_modelling_document_topic + '/processed.csv'
        df = pd.read_csv(path, dtype={
            'hash_id': str,
            'topic_id': int})

        df = self.df_ref_hash_id.merge(df, left_on='hash_id', right_on='hash_id')

        print('Raw Input Specifications')
        print(df.head())
        print(df.columns)
        print(df.dtypes)
        print(df.shape)

        table_name = 'pub_document_topic'
        self.drop_table(table_name)

        df_chunked = [df[i:i + n_chunk_size] for i in range(0, df.shape[0], n_chunk_size)]
        for idx, c in enumerate(df_chunked):
            mode = self.set_db_update_mode(idx)
            self.update_db(c, table_name, mode)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'hash_id')
            conn.execute(sql_string)

        print('Updated table {}'.format(table_name))

    def update_db_from_authors_by_paper(self):
        print('--- Update Db From Authors By Paper')
        n_chunk_size = 10000  # chunk row size

        path = self.args.input_authors_by_paper + '/processed.csv'
        df = pd.read_csv(path, dtype={
            'paper_id': str,
            'hash_id': str,
            'author': str,
            'hash_author': str})

        print('Raw Input Specifications')
        print(df.head())
        print(df.columns)
        print(df.dtypes)
        print(df.shape)

        table_name = 'pub_authors_by_paper'
        self.drop_table(table_name)

        df = df[['hash_id', 'hash_author']]
        df_chunked = [df[i:i + n_chunk_size] for i in range(0, df.shape[0], n_chunk_size)]
        for idx, c in enumerate(df_chunked):
            mode = self.set_db_update_mode(idx)
            self.update_db(c, table_name, mode)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'hash_id, hash_author')
            conn.execute(sql_string)

        print('Updated table {}'.format(table_name))

    def update_db_from_author_article_metrics(self):
        print('--- Update Db From Author Article Metrics')
        n_chunk_size = 10000  # chunk row size

        path = self.args.input_author_article_metrics + '/processed.csv'
        df = pd.read_csv(path, dtype={
            'hash_id': str,
            'paper_id': str,
            'title': str,
            'pubmed_id': str,
            'publish_year': int,
            'publish_time': str,
            'author_count': float,
            'referenced_papers_count': float,
            'paper_citation_count': float,
            'paper_citation_count_norm': float,
            'paper_pagerank': float,
            'author_pagerank_aggr': float,
            'author_citation_ratio_sum': float,
            'score_mf1': float,
            'score_mf2': float,
            'score_mf3': float,
            'score_mf4': float
        })

        print('Raw Input Specifications')
        print(df.head())
        print(df.columns)
        print(df.dtypes)
        print(df.shape)

        table_name = 'pub_paper_metrics'
        self.drop_table(table_name)

        df = df[['hash_id', 'author_count', 'referenced_papers_count', 'paper_citation_count',
                 'paper_citation_count_norm', 'paper_pagerank', 'author_pagerank_aggr', 'author_citation_ratio_sum',
                 'score_mf1', 'score_mf2', 'score_mf3', 'score_mf4']]

        df_chunked = [df[i:i + n_chunk_size] for i in range(0, df.shape[0], n_chunk_size)]
        for idx, c in enumerate(df_chunked):
            mode = self.set_db_update_mode(idx)
            self.update_db(c, table_name, mode)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'hash_id')
            conn.execute(sql_string)

        print('Updated table {}'.format(table_name))

        print('Setting master hash id list')
        self.df_ref_hash_id = df[['hash_id']].copy()
        print('Master hash id list has {} articles'.format(str(len(self.df_ref_hash_id))))

    def update_db_from_trials(self):
        print('--- Update Db From Trials')
        n_chunk_size = 10000  # chunk row size

        path = self.args.input_trials + '/processed.csv'
        df = pd.read_csv(path, dtype={
            'hash_id': str,
            'id': str})

        print('Raw Input Specifications')
        print(df.head())
        print(df.columns)
        print(df.dtypes)
        print(df.shape)

        table_name = 'pub_trials'
        self.drop_table(table_name)

        df_chunked = [df[i:i + n_chunk_size] for i in range(0, df.shape[0], n_chunk_size)]
        for idx, c in enumerate(df_chunked):
            mode = self.set_db_update_mode(idx)
            self.update_db(c, table_name, mode)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'hash_id, id')
            conn.execute(sql_string)

        print('Updated table {}'.format(table_name))

    def update_db(self, df, table_name, mode):
        df.to_sql(table_name, self.sqlalchemy_engine, index=False, method='multi', if_exists=mode)

    def drop_table(self, table_name):
        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'DROP TABLE {}'.format(table_name)
            try:
                conn.execute(sql_string)
            except exc.SQLAlchemyError:
                print('Exception dropping {}'.format(table_name))

    def collect_metrics_pre(self):
        pass
        #self.articles_in = len(self.df)

    def log_metrics_post(self):
        pass


if __name__ == "__main__":
    print('--- Database Update')
    database_update = DatabaseUpdate()
    print('--- Database Update Completed')
