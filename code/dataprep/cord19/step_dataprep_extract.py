"""
Extract raw text from the input data, parsing the JSON and identifying the required information
"""
import numpy as np
import pandas as pd
import glob
import json
import argparse
import os
from azureml.core import Run
from matplotlib import pyplot as plt
import seaborn as sns


class Dummy(object):
    pass


class Extract:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.df_meta = None
        self.df = pd.DataFrame()
        self.df_preprint = pd.DataFrame()

        self.articles_in = 0

        self.get_runtime_arguments()

        self.load_dataset_metadata()
        self.load_pubmed_data()
        self.load_preprint_data()
        self.merge()

        self.collect_metrics_pre()

        self.cleanse()

        self.log_metrics_post()

        self.output_dataset()

    @staticmethod
    def article_read(file_path):
        article = Dummy()
        with open(file_path) as file:
            content = json.load(file)
            article.paper_id = content['paper_id']
            article.body_text = []
            article.abstract = []
            article.results = []
            article.bibliography = []

            # Abstract
            if 'abstract' in content:
                for entry in content['abstract']:
                    article.abstract.append(entry['text'])

            # Body text
            for entry in content['body_text']:
                article.body_text.append(entry['text'])

            # Results
            results_synonyms = ['result']
            for entry in content['body_text']:
                section_title = ''.join(x.lower() for x in entry['section'] if x.isalpha())
                if any(r in section_title for r in results_synonyms):
                    article.results.append(entry['text'])

            # Bibliography
            # for entry in content['bib_entries']:
            for key, value in content['bib_entries'].items():
                bib = {'ref': key, 'title': str(value['title']).lower(), 'venue': str(value['venue']),
                       'year': str(value['year'])}
                article.bibliography.append(str(bib))

            article.abstract = '\n'.join(article.abstract)
            article.body_text = '\n'.join(article.body_text)
            article.results = '\n'.join(article.results)
            article.bibliography = '\n'.join(article.bibliography)

            return article

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input',
            type=str,
            help='Path to input data'
        )
        parser.add_argument(
            '--dataset_size_percent',
            type=int,
            help='Percentage of raw dataset to process'
        )
        parser.add_argument(
            '--pmc_only',
            type=str,
            help='Process only PMC dataset'
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Cleansed directory'
        )

        self.args = parser.parse_args()

        print('Input: {}'.format(self.args.input))
        print('Dataset size Percent: {}'.format(str(self.args.dataset_size_percent)))
        print('process only PMC dataset: {}'.format(str(self.args.pmc_only)))
        print('Output: {}'.format(self.args.output))

    def load_dataset_metadata(self):
        print('--- Get Dataset Metadata')
        metadata_path = f'{self.args.input}/metadata.csv'
        self.df_meta = pd.read_csv(metadata_path, dtype={
            'cord_uid': str,
            'sha': str,
            'source_x': str,
            'title': str,
            'doi': str,
            'pmcid': str,
            'pubmed_id': str,
            'license': str,
            'abstract': str,
            'publish_time': str,
            'authors': str,
            'journal': str,
            'mag_id': str,
            'who_covidence_id': str,
            'arxiv_id': str,
            'pdf_json_files': str,
            'pubmed_json_files': str,
            'url': str,
            's2_id': str
        })

        print('Dataset raw size ', str(len(self.df_meta)))

        print('\nMetadata head')
        print(self.df_meta.head(2))

        print('\nMissing value count by column')
        print(self.df_meta.isnull().sum())

        print('\nUnique uid count:', self.df_meta.cord_uid.nunique())
        print('\nUnique sha count:', self.df_meta.sha.nunique())
        print('\nUnique title count:', self.df_meta.title.nunique())

    def load_pubmed_data(self):
        print('--- Get PMC Articles')

        pubmed_article_list = glob.glob(f'{self.args.input}/**/pmc_json/*.json', recursive=True)
        print('Number Pubmed articles {}'.format(str(len(pubmed_article_list))))

        subset = int(len(pubmed_article_list) * (self.args.dataset_size_percent / 100))
        pubmed_article_list = pubmed_article_list[:subset]
        print('Number Pubmed articles after {} percent subset  {}'.format(str(self.args.dataset_size_percent),
                                                                          str(len(pubmed_article_list))))

        pubmed_articles = {'paper_id': [], 'body_text': [], 'methods': [], 'results': [], 'bibliography': []}
        for idx, entry in enumerate(pubmed_article_list):
            if idx % (len(pubmed_article_list) // 10) == 0:
                print(f'Processing index: {idx} of {len(pubmed_article_list)}')
            content = self.article_read(entry)
            pubmed_articles['paper_id'].append(content.paper_id)
            pubmed_articles['body_text'].append(content.body_text)
            pubmed_articles['results'].append(content.results)
            pubmed_articles['bibliography'].append(content.bibliography)

        df_pubmed_articles = pd.DataFrame(pubmed_articles, columns=['paper_id', 'body_text', 'results', 'bibliography'])
        df_pubmed_articles['subset_source'] = 'PMC'
        df_pubmed_articles.head()

        print('Pubmed text shape {}'.format(df_pubmed_articles.shape))

        self.df = pd.merge(df_pubmed_articles, self.df_meta, left_on='paper_id', right_on='pmcid', how='left').drop('pmcid', axis=1)

        print(self.df.columns)
        print(self.df.head(3))
        print(self.df.isnull().sum())

    def load_preprint_data(self):
        print('--- Get Preprint Articles')

        if self.args.pmc_only == 'y':
            print('Excluding preprint load - PMC only')
            return

        preprint_article_list = glob.glob(f'{self.args.input}/**/pdf_json/*.json', recursive=True)
        print('Number Preprint articles {}'.format(str(len(preprint_article_list))))

        subset = int(len(preprint_article_list) * (self.args.dataset_size_percent / 100))
        preprint_article_list = preprint_article_list[:subset]
        print('Number Preprint articles after {} percent subset  {}'.format(str(self.args.dataset_size_percent),
                                                                            str(len(preprint_article_list))))

        preprint_articles = {'paper_id': [], 'abstract': [], 'body_text': [], 'results': [], 'bibliography': []}
        for idx, entry in enumerate(preprint_article_list):
            if idx % (len(preprint_article_list) // 10) == 0:
                print(f'Processing index: {idx} of {len(preprint_article_list)}')
            content = self.article_read(entry)
            preprint_articles['paper_id'].append(content.paper_id)
            preprint_articles['abstract'].append(content.abstract)
            preprint_articles['body_text'].append(content.body_text)
            preprint_articles['results'].append(content.results)
            preprint_articles['bibliography'].append(content.bibliography)

        df_preprint_articles = pd.DataFrame(preprint_articles, columns=['paper_id', 'abstract', 'body_text', 'results',
                                                                        'bibliography'])
        df_preprint_articles.head()

        print('Preprint text shape {}'.format(df_preprint_articles.shape))

        self.df_preprint = pd.merge(df_preprint_articles, self.df_meta, left_on='paper_id', right_on='sha', how='left')
        self.df_preprint['subset_source'] = 'Preprint'

        print(self.df_preprint.columns)
        print(self.df_preprint.head(3))
        print(self.df_preprint.isnull().sum())

    def merge(self):
        print('--- Merging PMC Peer-review & Pre-print Article Collection')
        try:
            self.df = pd.merge(left=self.df, right=self.df_preprint, how='outer')
        except pd.errors.MergeError as e:
            print('Merging error, likely due to PMC-only subset')

        print('Number articles after merge:', str(len(self.df)))

    def cleanse(self):
        print('--- Cleansing Article Collection')

        print('\nNon-empty column value count')
        print(self.df.count())

        print('\nUnique uid count:', self.df.cord_uid.nunique())
        print('\nUnique sha count:', self.df.sha.nunique())
        print('\nUnique title count:', self.df.title.nunique())

        # Remove articles with no title originating from meta
        print('Removing articles with no title')
        self.df = self.df[self.df['title'].notna()]

        print('\nUnique uid count:', self.df.cord_uid.nunique())
        print('\nUnique sha count:', self.df.sha.nunique())
        print('\nUnique title count:', self.df.title.nunique())

        print('Removing latext commands')
        re_ids = [r'\\usepackage.*}', r'\\setlength.*}', '{.*}', '{.*pt}', r'\\begin{.*}', r'\\document.*}', 'left(.*)']
        text_re_ids = ('|').join(re_ids)
        self.df['title'] = self.df['title'].str.replace(text_re_ids, '', regex=True)
        self.df['abstract'] = self.df['abstract'].str.replace(text_re_ids, '', regex=True)
        self.df['body_text'] = self.df['body_text'].str.replace(text_re_ids, '', regex=True)

        print('Populating empty abstracts')
        try:
            self.df['abstract'].fillna(self.df['abstract_x'], inplace=True)
            self.df['abstract'].fillna(self.df['abstract_y'], inplace=True)
            self.df.loc[self.df.abstract == '', 'abstract'] = self.df['abstract_x']
            self.df.loc[self.df.abstract == '', 'abstract'] = self.df['abstract_y']
        except KeyError:
            print('KeyError copying extract, possibly due to PMC-only subset')

        print('\nNon-empty column value count')
        print(self.df.count())

        print('Removing articles with no abstract or body text')
        self.df = self.df[self.df['body_text'].notna()]
        self.df = self.df[self.df['abstract'].notna()]
        self.df = self.df[self.df.body_text != '']
        self.df = self.df[self.df.abstract != '']

        print('Removing articles with no publish time')
        self.df = self.df[self.df['publish_time'].notna()]

        # Replace nan with empty sting
        self.df.fillna('', inplace=True)

        # Drop unwanted columns
        self.df.drop(columns=['license', 'mag_id', 'who_covidence_id', 'arxiv_id', 'pdf_json_files', 'pmc_json_files',
                              's2_id'], inplace=True)

        try:
            self.df.drop(columns=['pmcid', 'abstract_x', 'abstract_y'], inplace=True)
        except KeyError:
            print('KeyError copying extract, possibly due to PMC-only subset')

        self.df.rename(columns={'source_x': 'source'}, inplace=True)

        print('Cleansing article titles')
        self.df['title'] = self.df['title'].str.strip('\n')
        self.df['title'] = self.df['title'].str.replace('[^\w\s]', '', regex=True)
        self.df['title'].replace('', np.nan, inplace=True)
        self.df = self.df[self.df['title'].notna()]

        print('Number duplicated titles {}'.format(self.df[self.df.duplicated(subset=['title'], keep=False)].shape[0]))
        print('Number duplicated cord_uid {}'.format(self.df[self.df.duplicated(subset=['cord_uid'],
                                                                                keep=False)].shape[0]))

        print('Dropping duplicates')
        # Sort descending for PMC by latest date then preprint by latest date
        self.df.sort_values(by=['cord_uid', 'subset_source', 'publish_time'], ascending=[True, True, False],
                            inplace=True)
        self.df.drop_duplicates(['cord_uid'], keep='first', inplace=True)
        self.df.drop_duplicates(['title'], keep='first', inplace=True)

        self.df.fillna('', inplace=True)
        self.df = self.df.replace(np.nan, '')

        print('Dropping articles with empty body text')
        self.df['body_text'] = self.df['body_text'].apply(str)
        self.df = self.df[(self.df['body_text'] != '') & (self.df['title'] != '')]

        print('\nParagraph stats before drop')
        self.df['body_paragraph'] = self.df['body_text'].str.split('\n', expand=False)
        self.df['body_paragraph_count'] = self.df['body_paragraph'].str.len()

        fig_name = 'Article Paragraph Count Distribution - Before Cleanse'
        fig, ax = plt.subplots(figsize=(32, 14))
        sns.distplot(self.df['body_paragraph_count'])
        plt.xlabel('Body Paragraph Count', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close()

        print('Number papers before paragraph check: {}'.format(self.df.shape[0]))
        print('Avg Article Paragraphs', self.df['body_paragraph'].str.len().mean())
        print('Min Article Paragraphs', self.df['body_paragraph'].str.len().min())
        print('Max Article Paragraphs', self.df['body_paragraph'].str.len().max())

        # Remove articles with outlier number of paragraphs
        p_threshold_min = 9
        p_threshold_max = 501
        self.df = self.df[self.df['body_paragraph'].str.len() > p_threshold_min]
        self.df = self.df[self.df['body_paragraph'].str.len() < p_threshold_max]
        self.df['body_paragraph_count'] = self.df['body_paragraph'].str.len()

        fig_name = 'Article Paragraph Count Distribution - After Cleanse'
        fig, ax = plt.subplots(figsize=(32, 14))
        sns.distplot(self.df['body_paragraph_count'])
        plt.xlabel('Body Paragraph Count', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close()

        print('Number papers after paragraph check: {}'.format(self.df.shape[0]))
        print('Avg Article Paragraphs', self.df['body_paragraph'].str.len().mean())
        print('Min Article Paragraphs', self.df['body_paragraph'].str.len().min())
        print('Max Article Paragraphs', self.df['body_paragraph'].str.len().max())

        self.df.drop(columns=['body_paragraph', 'body_paragraph_count'], inplace=True)

        self.df = self.df[self.df['body_text'].str.len() > 50]
        print('Number papers after drops and filters {}'.format(self.df.shape[0]))
        print('Number unique papers after drops and filters {}'.format(str(self.df.cord_uid.nunique())))
        print('\nNon-empty column value count')
        print(self.df.count())

    def collect_metrics_pre(self):
        self.articles_in = len(self.df)

    def log_metrics_post(self):
        self.run.log('# Articles In', self.articles_in)
        self.run.log('# Articles Out', len(self.df))

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


if __name__ == '__main__':
    print('--- Extract Started')
    extract = Extract()
    print('--- Extract Completed')
