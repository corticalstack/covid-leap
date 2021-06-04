import argparse
import os
from azureml.core import Run
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Discovery:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.df = None
        self.articles_in = 0

        self.get_runtime_arguments()

        self.load_dataset()

        self.collect_metrics_pre()
        self.discovery()
        self.log_metrics_post()

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input',
            type=str,
            help='Input extract data'
        )

        self.args = parser.parse_args()

        print('Input: {}'.format(self.args.input))

    def load_dataset(self):
        print('--- Load Data')
        path = self.args.input + '/processed.csv'
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
        print(self.df.head())
        print(self.df.columns)
        print(self.df.shape)

    def discovery(self):
        print('--- Discovery')
        self.articles_by_journal()

    def articles_by_journal(self):
        # Top journals by number of articles
        top_n = 10
        articles_by_journal = self.df['journal'].value_counts()
        df_temp = pd.DataFrame({'journal': articles_by_journal.index, 'Count': articles_by_journal.values})
        df_temp['journal'].replace('', np.nan, inplace=True)
        df_temp.dropna(subset=['journal'], inplace=True)
        df_temp.sort_values(by=['Count'], ascending=False, inplace=True)

        fig_name = 'Top journals by number of articles'
        fig, ax = plt.subplots(figsize=(32, 14))
        sns.barplot(y='journal', x='Count', data=df_temp[:top_n], palette='husl')
        plt.ylabel('Journal', fontsize=24)
        plt.xlabel('Articles', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        fig.subplots_adjust(left=0.2)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close()

        # Top reputable journals by number of articles
        reputable_journal_list = ['Nature', 'Lancet Infect Dis', 'The Lancet', 'Lancet', 'Lancet Glob Health',
                                  'Lancet Public Health', 'Emerg Infect Dis', 'BMJ', 'BMJ Glob Health', 'J Infect Dis']
        df_temp = df_temp.loc[df_temp['journal'].isin(reputable_journal_list)]
        df_temp.sort_values(by=['Count'], ascending=False, inplace=True)

        fig_name = 'Top reputable journals by number of articles'
        fig, ax = plt.subplots(figsize=(32, 14))
        sns.barplot(y='journal', x='Count', data=df_temp[:top_n], palette='husl')
        plt.ylabel('Journal', fontsize=24)
        plt.xlabel('Articles', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        fig.subplots_adjust(left=0.2)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close()

    def collect_metrics_pre(self):
        self.articles_in = len(self.df)

    def log_metrics_post(self):
        self.run.log('# Articles In', self.articles_in)

    def offline_save_fig(self, name):
        if 'OfflineRun' in self.run._identity:
            full_path = 'plots/' + name + '.png'
            plt.savefig(full_path, dpi=300, format='png')


if __name__ == "__main__":
    print('--- Discovery Started')
    discovery = Discovery()
    print('--- Discovery Completed')
