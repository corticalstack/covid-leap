import argparse
import os
from azureml.core import Run
import numpy as np
import pandas as pd


class Trials:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.trials_re_ids = None

        self.trials_per_article = []
        self.df = pd.DataFrame()

        self.articles_in = 0
        self.papers_referencing_trials = 0

        self.get_runtime_arguments()

        self.load_dataset()

        self.set_trials_re_ids()

        self.collect_metrics_pre()
        self.find_trials_in_articles()
        self.log_metrics_post()

        self.output_dataset()

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input',
            type=str,
            help='Input dataset'
        )
        parser.add_argument(
            '--output',
            type=str,
            help=' Output dataset'
        )
        self.args = parser.parse_args()

        print('Input Dataset: {}'.format(self.args.input))
        print('Output Dataset: {}'.format(self.args.output))

    def load_dataset(self):
        print('--- Load Dataset')
        path = f'{self.args.input}/processed.csv'
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
            'url': str,
            'hash_id': str})

        print('Raw Input Specifications')
        print(self.df.head())
        print(self.df.columns)
        print(self.df.shape)

    def set_trials_re_ids(self):
        re_id_nct = 'NCT[0-9]{8}'
        re_id_chi = 'ChiCTR[0-9]{10}'
        re_id_pac = 'PACTR[0-9]{15}'
        re_id_eu = 'EUCTR[0-9]{4}-[0-9]{6}-[0-9]{2}-[A-Z]{2}'
        re_id_isrctn = 'ISRCTN[0-9]{8}'
        re_id_per = 'PER-[0-9]{3}-[0-9]{2}'
        re_id_jrct = 'jRCTPER[0-9]{10}'
        re_id_actrn = 'ACTRN[0-9]{14}'
        re_id_ctri = 'CTRI/[0-9]{4}/[0-9]{2}/[0-9]{6}'
        re_id_rpce = 'RPCEC[0-9]{8}'

        re_ids = [re_id_nct, re_id_chi, re_id_pac, re_id_eu, re_id_isrctn, re_id_per, re_id_jrct, re_id_actrn, 
                  re_id_ctri, re_id_rpce]

        self.trials_re_ids = ('|').join(re_ids)
        self.trials_re_ids = r'({})'.format(self.trials_re_ids)

    def trials_to_vertical(self, row):
        for x in row[1]:
            self.trials_per_article.append([row[0], x])

    def find_trials_in_articles(self):
        # Find all occurrences of the regexpr containing trial ID notation
        # Create series, dumping located trial ID's into list
        s_trials = (self.df.title.fillna('') + ' ' + self.df.abstract.fillna('') + ' ' +
                    self.df.body_text.fillna('') + self.df.results.fillna(''))\
            .str.findall(self.trials_re_ids)

        # Count number of located trials in articles
        self.papers_referencing_trials = sum([len(x) != 0 for x in s_trials])
        print('Number of trials found in papers: {}'.format(str(self.papers_referencing_trials)))

        df_trials = pd.DataFrame(s_trials, columns=['trials'])
        df_trials.set_index(self.df.hash_id, inplace=True)
        df_trials = df_trials[df_trials.trials.str.len() != 0]

        df_trials.reset_index(inplace=True)
        self.df = self.df.merge(df_trials, how='left', on='hash_id').fillna('')
        self.df = self.df[['hash_id', 'trials']]

        self.df['dummy'] = self.df.apply(lambda row: self.trials_to_vertical(row), axis=1)
        self.df = pd.DataFrame(self.trials_per_article, columns=['hash_id', 'id'])

        self.df.drop_duplicates(subset=['hash_id', 'id'], keep='first', inplace=True)

    def collect_metrics_pre(self):
        self.articles_in = len(self.df)

    def log_metrics_post(self):
        self.run.log('# Articles In', self.articles_in)
        self.run.log('# Articles Out', len(self.df))
        self.run.log('# Papers Referencing Trials', self.papers_referencing_trials)

    def output_dataset(self):
        print('--- Output Dataset')
        if not (self.args.output is None):
            os.makedirs(self.args.output, exist_ok=True)
            path = self.args.output + '/processed.csv'
            self.df.to_csv(path, index=False)
            print('Output created: {}'.format(path))
            print('Column definition of output')
            print(self.df.columns)


if __name__ == '__main__':
    print('--- Trials')
    trials = Trials()
    print('--- Trials Completed')
