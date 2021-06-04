import argparse
import os
from azureml.core import Run
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class FeatureEngineering:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.df = None
        self.articles_in = 0

        self.get_runtime_arguments()

        self.load_dataset()

        self.collect_metrics_pre()

        self.feature_engineering()

        self.plot_disease_category()
        self.log_metrics_post()

        self.output_dataset()

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

    def feature_engineering(self):
        print('--- Feature Engineering')

        # Publish Year
        self.set_publish_year()

        # Disease category
        self.set_is_coronavirus()
        self.set_is_sars_cov2()
        self.set_is_sars_cov()
        self.set_is_mers()

    def set_publish_year(self):
        print('--- Setting publish year')
        self.df['publish_year'] = self.df.publish_time.str[:4].fillna(-1).astype(int)  # 360 times None

    def set_is_coronavirus(self):
        print('--- Setting is_coronavirus')
        # Use regex with \b word boundary to find full words, including multi-word strings
        regexpr = r'\bcovid\b|\bcovid 19\b|\bcovid-19\b|\b2019-nCoV\b|\bcoronavirus\b|\bsars cov\b|\bsars-cov\b|'\
                  r'\bsars-cov-2\b|\bmers\b|\bmers-cov\b|\bmiddle east respiratory syndrome\b|'\
                  r'\bsevere acute respiratory syndrome\b|\bhcov'
        self.df['is_coronavirus'] = self.df.body_text.str.contains(regexpr, regex=True, case=False)
        self.df['is_coronavirus_title'] = self.df.title.str.contains(regexpr, regex=True, case=False)

    def set_is_sars_cov2(self):
        print('--- Setting is_sars_cov2')
        regexpr = r'\bcovid 19\b|\bcovid-19\b|\b2019-nCoV\b|\bsars-cov-2'
        self.df['is_sars_cov2'] = self.df.body_text.str.contains(regexpr, regex=True, case=False)
        self.df['is_sars_cov2_title'] = self.df.title.str.contains(regexpr, regex=True, case=False)

    def set_is_sars_cov(self):
        print('--- Setting is_sars_cov')
        regexpr = r'sars-cov'
        self.df['is_sars_cov'] = self.df.body_text.str.contains(regexpr, regex=True, case=False)
        self.df['is_sars_cov_title'] = self.df.title.str.contains(regexpr, regex=True, case=False)

    def set_is_mers(self):
        print('--- Setting is_mers')
        regexpr = r'\bmers\b|\bmers-cov\b|\bmiddle east respiratory syndrome\b'
        self.df['is_mers'] = self.df.body_text.str.contains(regexpr, regex=True, case=False)
        self.df['is_mers_title'] = self.df.title.str.contains(regexpr, regex=True, case=False)

    def collect_metrics_pre(self):
        self.articles_in = len(self.df)

    def plot_disease_category(self):
        print('--- Plot disease cat')
        x_ticks = 15

        print('Publication Body Text by Disease Category')
        df_disease_cat = self.df.groupby(['publish_year'])[['is_coronavirus', 'is_sars_cov2', 'is_sars_cov',
                                                            'is_mers']].sum()
        df_disease_cat.rename(columns={'is_coronavirus': 'Coronavirus', 'is_sars_cov': 'Sars-CoV', 'is_sars_cov2':
            'Sars-CoV-2', 'is_mers': 'Mers-CoV'}, inplace=True)
        df_disease_cat.reset_index(inplace=True)
        df_disease_cat.sort_values(by=['publish_year'], inplace=True)
        print(df_disease_cat)

        fig_name = 'Publication Body Text by Disease Category'
        fig, ax = plt.subplots(figsize=(26, 13))
        sns.lineplot(x='publish_year', y='value', hue='variable', linewidth=2, data=pd.melt(df_disease_cat,
                                                                                            ['publish_year']))
        plt.xlabel('Publish Year', fontsize=24)
        plt.ylabel('Articles', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        leg = ax.legend()
        for line in leg.get_lines():
            line.set_linewidth(2.0)
        ax.legend(title='Disease', fontsize=20, title_fontsize=24)

        ax.xaxis.set_major_locator(MaxNLocator(x_ticks))
        self.run.log_image(fig_name, plot=fig)

        self.offline_save_fig(fig_name)
        plt.close()

        print('Publication Title by Disease Category')
        df_disease_cat = self.df.groupby(['publish_year'])[['is_coronavirus_title', 'is_sars_cov2_title',
                                                            'is_sars_cov_title', 'is_mers_title']].sum().astype(int)
        df_disease_cat.rename(columns={'is_coronavirus_title': 'Coronavirus', 'is_sars_cov_title': 'Sars-CoV',
                                       'is_sars_cov2_title': 'Sars-CoV-2', 'is_mers_title': 'Mers-CoV'}, inplace=True)
        df_disease_cat.reset_index(inplace=True)

        df_disease_cat.sort_values(by=['publish_year'], inplace=True)
        print(df_disease_cat)

        fig_name = 'Publication Title by Disease Category'
        fig, ax = plt.subplots(figsize=(26, 13))
        sns.lineplot(x='publish_year', y='value', hue='variable', linewidth=2, data=pd.melt(df_disease_cat,
                                                                                            ['publish_year']))
        plt.xlabel('Publish Year', fontsize=24)
        plt.ylabel('Articles', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        leg = ax.legend()
        for line in leg.get_lines():
            line.set_linewidth(2.0)
        ax.legend(title='Disease', fontsize=20, title_fontsize=24)

        ax.xaxis.set_major_locator(MaxNLocator(x_ticks))
        self.run.log_image(fig_name, plot=fig)

        self.offline_save_fig(fig_name)
        plt.close()

    def log_metrics_post(self):
        self.run.log('# Articles In', self.articles_in)
        self.run.log('# Articles Out', len(self.df))
        self.run.log('# Articles Coronavirus', self.df.is_coronavirus.sum())
        self.run.log('# Articles Coronavirus (title)', self.df.is_coronavirus_title.sum())
        self.run.log('# Articles Sars-CoV-2', self.df.is_sars_cov2.sum())
        self.run.log('# Articles Sars-CoV-2 (title)', self.df.is_sars_cov2_title.sum())
        self.run.log('# Articles Sars-CoV', self.df.is_sars_cov.sum())
        self.run.log('# Articles Sars-CoV (title)', self.df.is_sars_cov_title.sum())
        self.run.log('# Articles Mers', self.df.is_mers.sum())
        self.run.log('# Articles Mers (title)', self.df.is_mers_title.sum())

    def offline_save_fig(self, name):
        if 'OfflineRun' in self.run._identity:
            full_path = 'plots/' + name + '.png'
            plt.savefig(full_path, dpi=300, format='png')

    def output_dataset(self):
        print('--- Output Dataset')
        if not (self.args.output is None):
            os.makedirs(self.args.output, exist_ok=True)
            path = self.args.output + '/processed.csv'
            self.df.to_csv(path, index=False)
            print('Output created: {}'.format(path))
            print('Column definition of output')
            print(self.df.columns)


if __name__ == "__main__":
    print('--- Feature Engineering Started')
    feature_engineering = FeatureEngineering()
    print('--- Feature Engineering Completed')

