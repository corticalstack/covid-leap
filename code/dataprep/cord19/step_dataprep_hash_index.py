import argparse
import os
from azureml.core import Run
import numpy as np
import pandas as pd
import hashlib


class HashIndex:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.df = None
        self.articles_in = 0

        self.get_runtime_arguments()
        self.load_dataset()

        self.collect_metrics_pre()
        self.set_hash()
        self.log_metrics_post()

        self.output_dataset()

    @staticmethod
    def hash(row):
        row_hash = row['title'].lower().encode('utf-8')
        row_hash = hashlib.sha1(row_hash).hexdigest()
        return row_hash

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input',
            type=str,
            help='Input extract data'
        )

        parser.add_argument(
            '--output',
            type=str,
            help=' Output extract data'
        )

        self.args = parser.parse_args()

        print('Input: {}'.format(self.args.input))
        print('Output: {}'.format(self.args.output))

    def load_dataset(self):
        print('--- Load Data')
        path = self.args.input + "/processed.csv"
        self.df = pd.read_csv(path, dtype={
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
            'url': str})

        print('Raw Input Specifications')
        print(self.df.head())
        print(self.df.columns)
        print(self.df.shape)

    def set_hash(self):
        print('--- Creating Hash Index')
        self.df['hash_id'] = self.df.apply(lambda row: self.hash(row), axis=1)
        self.df.drop_duplicates(['hash_id'], inplace=True)

    def collect_metrics_pre(self):
        self.articles_in = len(self.df)

    def log_metrics_post(self):
        self.run.log('# Articles In', self.articles_in)
        self.run.log('# Articles Out', len(self.df))

    def output_dataset(self):
        print('--- Output Dataset')
        if not (self.args.output is None):
            os.makedirs(self.args.output, exist_ok=True)
            path = self.args.output + "/processed.csv"
            self.df.to_csv(path, index=False)
            print('Output created: {}'.format(path))
            print('Column definition of output')
            print(self.df.columns)


if __name__ == "__main__":
    print('--- Create hash index')
    hash_index = HashIndex()
    print('--- Hash index Completed')
