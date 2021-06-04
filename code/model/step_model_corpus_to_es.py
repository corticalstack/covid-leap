import argparse
import os
import re
import pandas as pd
import torch
import glob
from azureml.core import Run
from elasticsearch import Elasticsearch, helpers
import joblib
import tqdm.autonotebook


class CorpusEs:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.get_runtime_arguments()

        self.df = pd.DataFrame()

        self.path_corpus_as_df_chunk = 'corpus_as_df_chunk*.pkl'

        self.es_index_name = 'pub_text'
        self.es_conn = None
        self.es_host = None
        self.es_user = None
        self.es_pwd = None

        self.get_os_environ_variables()

        self.df_chunk_pkl = []
        self.get_df_chunks()

        self.set_es_conn()
        self.delete_es_index()

        for c in self.df_chunk_pkl:
            self.load_corpus_from_df(c)
            self.transform_for_es()
            self.load_corpus_into_es()

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--wrkdir',
            type=str,
            help='Model working directory'
        )
        parser.add_argument(
            '--es_batch_chunk_size',
            type=int,
            help='ElasticSearch batch chunk size'
        )

        self.args = parser.parse_args()

        print('Working Directory: {}'.format(self.args.wrkdir))
        print('ElasticSearch batch chunk size: {}'.format(self.args.es_batch_chunk_size))

    def get_os_environ_variables(self):
        print('--- Getting OS Environment Variables')
        self.es_host = os.environ.get('AZ_RP_ES_HOST')
        self.es_user = os.environ.get('AZ_RP_ES_USER')
        self.es_pwd = os.environ.get('AZ_RP_ES_PWD')

    def get_df_chunks(self):
        self.df_chunk_pkl = glob.glob(f'{self.args.wrkdir}/' + self.path_corpus_as_df_chunk, recursive=True)

    def set_es_conn(self):
        print('--- Setting ElasticSearch Connection')
        self.es_conn = Elasticsearch([self.es_host], http_auth=(self.es_user, self.es_pwd), scheme="https", port=443)

    def delete_es_index(self):
        print('--- Delete ElasticSearch index')
        print('Deleting index {}'.format(self.es_index_name))
        self.es_conn.indices.delete(index=self.es_index_name, ignore=[400, 404])

    def load_corpus_from_df(self, c):
        print('--- Load corpus from df')
        print('Loading chunk {}'.format(c))
        self.df = joblib.load(c)

    def transform_for_es(self):
        self.df['is_coronavirus'] = self.df['is_coronavirus'].replace('False', 'false')
        self.df['is_coronavirus'] = self.df['is_coronavirus'].replace('True', 'true')

        self.df['is_coronavirus_title'] = self.df['is_coronavirus_title'].replace('False', 'false')
        self.df['is_coronavirus_title'] = self.df['is_coronavirus_title'].replace('True', 'true')

        self.df['is_sars_cov2'] = self.df['is_sars_cov2'].replace('False', 'false')
        self.df['is_sars_cov2'] = self.df['is_sars_cov2'].replace('True', 'true')

        self.df['is_sars_cov2_title'] = self.df['is_sars_cov2_title'].replace('False', 'false')
        self.df['is_sars_cov2_title'] = self.df['is_sars_cov2_title'].replace('True', 'true')

        self.df['is_sars_cov'] = self.df['is_sars_cov'].replace('False', 'false')
        self.df['is_sars_cov'] = self.df['is_sars_cov'].replace('True', 'true')

        self.df['is_sars_cov_title'] = self.df['is_sars_cov_title'].replace('False', 'false')
        self.df['is_sars_cov_title'] = self.df['is_sars_cov_title'].replace('True', 'true')

        self.df['is_mers'] = self.df['is_mers'].replace('False', 'false')
        self.df['is_mers'] = self.df['is_mers'].replace('True', 'true')

        self.df['is_mers_title'] = self.df['is_mers_title'].replace('False', 'false')
        self.df['is_mers_title'] = self.df['is_mers_title'].replace('True', 'true')

    def load_corpus_into_es(self):
        print('--- Upload corpus into ELasticSearch')

        vector_dimensions = len(self.df.iloc[0]['paragraph_model_embedding'])
        print('Setting ES dense vector dimensions to {}'.format(str(vector_dimensions)))

        try:
            es_index = {
                'mappings': {
                    'properties': {
                        'hash_id': {
                          'type': 'text'
                        },
                        'title': {
                          'type': 'text'
                        },
                        'source': {
                          'type': 'text'
                        },
                        'doi': {
                          'type': 'text'
                        },
                          'pubmed_id': {
                              'type': 'text'
                          },
                          'journal': {
                              'type': 'text'
                          },
                          'url': {
                              'type': 'text'
                          },
                          'publish_year': {
                              'type': 'integer'
                          },
                          'topic_id': {
                              'type': 'integer'
                          },
                          'is_coronavirus': {
                              'type': 'boolean'
                          },
                          'is_coronavirus_title': {
                              'type': 'boolean'
                          },
                          'is_sars_cov2': {
                              'type': 'boolean'
                          },
                          'is_sars_cov2_title': {
                              'type': 'boolean'
                          },
                          'is_sars_cov': {
                              'type': 'boolean'
                          },
                          'is_sars_cov_title': {
                              'type': 'boolean'
                          },
                          'is_mers': {
                              'type': 'boolean'
                          },
                          'is_mers_title': {
                              'type': 'boolean'
                          },
                          'author_count': {
                              'type': 'integer'
                          },
                          'paper_citation_count': {
                              'type': 'integer'
                          },
                          'paper_pagerank': {
                              'type': 'float'
                          },
                          'score_mf1': {
                              'type': 'float'
                          },
                          'score_mf2': {
                              'type': 'float'
                          },
                          'score_mf3': {
                              'type': 'float'
                          },
                          'score_mf4': {
                              'type': 'float'
                          },
                          'text': {
                              'type': 'text'
                          },
                          'text_processed': {
                              'type': 'text'
                          },
                          'text_processed_vector': {
                              'type': 'dense_vector',
                              'dims': vector_dimensions
                          }
                    }
                }
            }

            self.es_conn.indices.create(index=self.es_index_name, body=es_index, ignore=[400])

            with tqdm.tqdm(total=len(self.df)) as pbar:
                for start_idx in range(0, len(self.df), self.args.es_batch_chunk_size):
                    end_idx = start_idx + self.args.es_batch_chunk_size

                    bulk_data = []
                    for index, row in self.df[start_idx:end_idx].iterrows():
                        bulk_data.append({
                                '_index': self.es_index_name,
                                '_id': row['id'],
                                '_source': {
                                    'hash_id': row['hash_id'],
                                    'title': row['title'],
                                    'source': row['source'],
                                    'doi': row['doi'],
                                    'pubmed_id': row['pubmed_id'],
                                    'journal': row['journal'],
                                    'url': row['url'],
                                    'publish_year': row['publish_year'],
                                    'topic_id': row['topic_id'],
                                    'is_coronavirus': row['is_coronavirus'],
                                    'is_coronavirus_title': row['is_coronavirus_title'],
                                    'is_sars_cov2': row['is_sars_cov2'],
                                    'is_sars_cov2_title': row['is_sars_cov2_title'],
                                    'is_sars_cov': row['is_sars_cov'],
                                    'is_sars_cov_title': row['is_sars_cov_title'],
                                    'is_mers': row['is_mers'],
                                    'is_mers_title': row['is_mers_title'],
                                    'author_count': row['author_count'],
                                    'paper_citation_count': row['paper_citation_count'],
                                    'paper_pagerank': row['paper_pagerank'],
                                    'score_mf1': row['score_mf1'],
                                    'score_mf2': row['score_mf2'],
                                    'score_mf3': row['score_mf3'],
                                    'score_mf4': row['score_mf4'],
                                    'text': row['paragraph'],
                                    'text_processed': row['paragraph_processed'],
                                    'text_processed_vector': row['paragraph_model_embedding']
                                }
                            })

                    helpers.bulk(self.es_conn, bulk_data)
                    pbar.update(self.args.es_batch_chunk_size)

        except Exception as e:
            print("During indexing an exception {} occurred. Continue\n\n".format(e))


if __name__ == '__main__':
    print('--- Corpus to ElasticSearch started')
    corpus_es = CorpusEs()
    print('--- Corpus to ElasticSearch completed')
