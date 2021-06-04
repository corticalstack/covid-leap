import argparse
import os
from azureml.core import Run
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import ast
import math as math
import hashlib
import datetime
from sklearn import preprocessing
import networkx as nx
import gc

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)


class AuthorArticleMetrics:
    def __init__(self):
        self.run = Run.get_context()
        self.current_year = datetime.datetime.now().year
        self.args = None
        self.df = None
        self.calculate_initial_metrics = False
        self.articles_in = 0

        self.number_papers_to_plot = 10

        self.get_runtime_arguments()


        self.load_dataset()

        self.collect_metrics_pre()

        self.df_paper_info = pd.DataFrame(
            {'paper_id': self.df['paper_id'], 'hash_id': self.df['hash_id'],
             'publish_time': self.df['publish_time'],
             'publish_year': self.df['publish_year']})

        self.df_author_info = pd.DataFrame(columns=['hash_author', 'author'])

        self.df_authors_by_paper = None

        self.df_paper_author_count = None

        self.df_bibliographies = pd.DataFrame(
           columns=['paper_id', 'hash_id', 'title', 'referenced_paper_title'])

        self.df_author_paper_count = None

        self.df_citations_per_paper = None

        self.df_paper_citations_per_author = None

        self.df_citations = None

        self.df_co_authors = None

        self.df_metrics = None

        self.set_authors()

        self.set_papers()

        self.set_bibliography()

        self.set_citations()

        self.set_author_paper_count()

        #self.set_paper_citations_per_author()

        self.set_author_paper_citations()

        self.build_co_authors()

        self.set_author_pagerank()

        self.set_paper_pagerank()

        self.consolidate_metrics()

        #self.df_metrics.to_csv('test_authors_papers_metrics', index=False)

        self.calculate_composite_metrics()

        self.log_metrics_post()

        self.output_datasets()

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input',
            type=str,
            help='Input extract data'
        )
        parser.add_argument(
            '--output_author_article_metrics',
            type=str,
            help=' Output author article metrics'
        )
        parser.add_argument(
            '--output_author_info',
            type=str,
            help=' Output author info'
        )
        parser.add_argument(
            '--output_authors_by_paper',
            type=str,
            help=' Output authors by paper'
        )

        self.args = parser.parse_args()

        print('Input: {}'.format(self.args.input))
        print('Output Author Article Metrics: {}'.format(self.args.output_author_article_metrics))
        print('Output Author Info: {}'.format(self.args.output_author_info))
        print('Output Authors By Paper: {}'.format(self.args.output_authors_by_paper))

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
            'hash_id': str,
            'publish_year': int,
            'is_coronavirus': str,
            'is_sars_cov2': str,
            'is_sars_cov': str,
            'is_mers': str,
        })

        print('Raw Input Specifications')
        print(self.df.head())
        print(self.df.columns)
        print(self.df.dtypes)
        print(self.df.shape)

        print('Input Following Column Subset')
        self.df = self.df[['hash_id', 'paper_id', 'title', 'doi', 'pubmed_id', 'authors', 'bibliography', 'journal',
                           'publish_year', 'publish_time']].fillna('')
        #[:10000]
        print(self.df.head())
        print(self.df.columns)
        print(self.df.dtypes)
        print(self.df.shape)

    def set_authors(self):
        print('--- Set Authors')

        s_authors = self.df['authors'].str.split(';', expand=True).stack()
        s_authors.replace('', np.nan, inplace=True)
        s_authors.dropna(inplace=True)

        idx_authors = s_authors.index.get_level_values(0)

        self.df_authors_by_paper = self.df[['paper_id', 'hash_id', 'doi', 'pubmed_id', 'journal']].loc[idx_authors].copy()
        self.df_authors_by_paper['author'] = s_authors.values
        del s_authors

        self.df_authors_by_paper['author'] = self.df_authors_by_paper['author'].str.strip()
        self.df_authors_by_paper['author'] = self.df_authors_by_paper['author'].replace(['.'], '')
        self.df_authors_by_paper['author'] = self.df_authors_by_paper['author'].str.lower()

        # Create author hash using author name and journal for disambiguation
        self.df_authors_by_paper['hash_author'] = self.df_authors_by_paper.apply(
            lambda row: hashlib.sha1((row['author'] + row['journal']).encode()).hexdigest(), axis=1)

        self.df_authors_by_paper.drop_duplicates(subset=['hash_id', 'hash_author'], keep='first', inplace=True)

        self.df_author_info = self.df_authors_by_paper[['hash_author', 'author']].copy().drop_duplicates().\
            reset_index(drop=True)

        print('\nStats for authors_by_paper')
        print(self.df_authors_by_paper.head())
        print(self.df_authors_by_paper.columns)
        print(self.df_authors_by_paper.shape)

        print('\nStats for author info')
        print(self.df_author_info.head())
        print(self.df_author_info.columns)
        print(self.df_author_info.shape)

        gc.collect()

    def set_papers(self):
        print('--- Set Papers')
        self.df_paper_author_count = self.df_authors_by_paper.groupby('hash_id')['author'].count().reset_index().rename(
            columns={"author": "author_count"})
        self.df_paper_author_count = self.df_paper_author_count.merge(self.df[['hash_id', 'title', 'pubmed_id']],
                                                                      on='hash_id', how='left')
        self.df_paper_info = self.df_paper_info.merge(self.df_paper_author_count, on='hash_id')
        self.df_paper_info.sort_values(by=['author_count'], ascending=False, inplace=True)

        print('\nStats for paper info')
        print(self.df_paper_info.head())
        print(self.df_paper_info.columns)
        print(self.df_paper_info.shape)

    def set_bibliography(self):
        print('--- Set Bibliography')
        n_chunk_size = 10000  # chunk row size
        df_bibliography_chunked = [self.df[i:i + n_chunk_size] for i in range(0, self.df.shape[0], n_chunk_size)]

        print('Processing bibliography in {} chunks'.format(str(len(df_bibliography_chunked))))
        for idx, dfc in enumerate(df_bibliography_chunked):
            print('Processing chunk ', str(idx))
            s_pb = dfc['bibliography'].str.split('\n', expand=True).stack()
            idx_bibliography = s_pb.index.get_level_values(0)
            df_temp = dfc[['paper_id', 'hash_id', 'title']].loc[idx_bibliography].copy()
            df_temp['referenced_paper_title'] = s_pb.values
            self.df_bibliographies = self.df_bibliographies.append(df_temp)

        del df_bibliography_chunked
        gc.collect()

        # Convert to dict
        print('Converting references to dictionary')
        self.df_bibliographies['referenced_paper_title'] = self.df_bibliographies['referenced_paper_title'].apply(ast.literal_eval)

        print('Getting referenced paper year and title')
        self.df_bibliographies['year'] = [d.get('year') for d in self.df_bibliographies.referenced_paper_title]
        self.df_bibliographies['rpt'] = [d.get('title') for d in self.df_bibliographies.referenced_paper_title]

        print('Cleaning referenced titles')
        self.df_bibliographies['rpt'] = self.df_bibliographies['rpt'].str.strip('\n')
        self.df_bibliographies['rpt'] = self.df_bibliographies['rpt'].str.replace('[^\w\s]', '', regex=True)
        self.df_bibliographies['rpt'].replace('', np.nan, inplace=True)
        self.df_bibliographies.dropna(subset=['rpt'], inplace=True)

        print('Hashing referenced titles')
        self.df_bibliographies['referenced_hash_id'] = self.df_bibliographies.apply(
            lambda row: hashlib.sha1(row['rpt'].lower().encode('utf-8')).hexdigest(), axis=1)

        self.df_bibliographies.drop(columns=['referenced_paper_title'], axis=1, inplace=True)
        self.df_bibliographies.rename(columns={'rpt': 'referenced_paper_title'}, inplace=True)

        print('\nStats for bibliographies')
        print(self.df_bibliographies.head())
        print(self.df_bibliographies.columns)
        print(self.df_bibliographies.shape)

    def set_citations(self):
        print('--- Set Citations')
        self.df_citations_per_paper = pd.DataFrame(
            {'hash_id': self.df_bibliographies['referenced_hash_id'].value_counts().index,
             'paper_citation_count': self.df_bibliographies['referenced_hash_id'].value_counts().values})

        self.df_paper_info = self.df_paper_info.merge(self.df_citations_per_paper, on='hash_id', how='left').fillna(0)

        self.df_paper_info.sort_values(by=['paper_citation_count'], ascending=False, inplace=True)
        print('\nStats for bibliographies')
        print(self.df_paper_info.head())

        del self.df_citations_per_paper
        gc.collect()

    def set_author_paper_count(self):
        print('--- Set Author Paper Count')
        self.df_author_paper_count = self.df_authors_by_paper.groupby('hash_author')['hash_id'].count().reset_index().rename(
            columns={'hash_id': 'paper_count'})

        self.df_author_info = self.df_author_info.merge(self.df_author_paper_count, on='hash_author', how='left')
        self.df_author_info.sort_values(by=['paper_count'], ascending=False, inplace=True)

        print('Top authors by paper count')
        print(self.df_author_info.head())
        print('Bottom authors by paper count')
        print(self.df_author_info.tail())

        del self.df_author_paper_count
        gc.collect()

    def set_paper_citations_per_author(self):
        print('--- Set Paper Citations Per Author')
        # Failing here
        self.df_paper_citations_per_author = self.df_authors_by_paper.merge(
            self.df_bibliographies, left_on='hash_id', right_on='referenced_hash_id', how='left').rename(
            columns={'paper_id_x': 'paper_id', 'hash_id_x': 'hash_id',
                     'paper_id_y': 'referencing_paper_id'}).drop(
            columns=['hash_id_y', 'referenced_hash_id', 'referenced_paper_title'])

        self.df_paper_citations_per_author = self.df_paper_citations_per_author.groupby(
            'hash_author').referencing_paper_id.count().reset_index().rename(
            columns={'referencing_paper_id': 'author_citation_count'})

        self.df_author_info = self.df_author_info.merge(self.df_paper_citations_per_author, on='hash_author', how='left')
        self.df_author_info.sort_values(by=['author_citation_count', 'paper_count'], inplace=True)

        print('Top authors by paper citation')
        print(self.df_author_info.head())
        print('Bottom authors by paper count')
        print(self.df_author_info.tail())

        del self.df_paper_citations_per_author
        gc.collect()

    def set_author_paper_citations(self):
        print('--- Set Author Paper Citations')
        self.df_citations = self.df_author_info.merge(self.df_authors_by_paper, on='hash_author', how='left').drop(
            columns=['author_y']).rename(columns={'author_x': 'author'})

        self.df_citations = self.df_citations.merge(self.df_paper_info, on='hash_id', how='left').rename(
            columns={'paper_id_x': 'paper_id', 'pubmed_id_x': 'pubmed_id'}).drop(['paper_id_y', 'pubmed_id_y'], axis=1)

        self.df_citations['paper_author_citation_ratio'] = \
            self.df_citations['paper_citation_count'] / self.df_citations['author_count']

        self.df_citations['author_citation_ratio'] = self.df_citations.groupby('hash_author')['paper_author_citation_ratio']\
            .transform('sum')

        self.df_citations.drop(['hash_id', 'paper_author_citation_ratio', 'author_count', 'publish_time',
                                'publish_year'], axis=1, inplace=True)

        self.df_citations = self.df_citations.drop_duplicates().reset_index(drop=True)

        self.df_citations.sort_values(by=['author_citation_ratio'], ascending=False, inplace=True)
        print('Top author paper citations')
        print(self.df_citations.head())
        print('Bottom author paper citations')
        print(self.df_citations.tail())

    def build_co_authors(self):
        print('--- Build Co Authors')
        self.df_co_authors = self.df_citations[['hash_author', 'paper_id']].merge(self.df_citations[['hash_author', 'paper_id']],
                                                                             on='paper_id').rename(
            columns={'hash_author_x': 'hash_author', 'hash_author_y': 'hash_co_author'})

        self.df_co_authors = self.df_co_authors[self.df_co_authors['hash_author'] != self.df_co_authors['hash_co_author']]

    def set_author_pagerank(self):
        print('--- Set Author Pagerank')

        df_temp = self.df_co_authors.copy()
        df_temp.drop(columns=['paper_id'], axis=1, inplace=True)
        df_temp['weight'] = 1
        df_temp = df_temp.groupby(['hash_author', 'hash_co_author'])['weight'].sum().reset_index()

        print('Normalising author co-author weights')
        cols_to_normalize = ['weight']
        x = df_temp[cols_to_normalize].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_temp_scaled = pd.DataFrame(x_scaled, columns=cols_to_normalize, index=df_temp.index)
        df_temp[cols_to_normalize] = df_temp_scaled

        print('Building author graph')
        graph = nx.from_pandas_edgelist(df_temp, 'hash_author', 'hash_co_author', ['weight'])

        print('Building author pagerank')
        author_pagerank = nx.pagerank(graph, tol=1e-9, alpha=0.8, dangling=None)
        df_author_pagerank = pd.DataFrame.from_dict(author_pagerank, orient='index')
        df_author_pagerank['hash_author'] = df_author_pagerank.index
        df_author_pagerank.columns = ['pagerank', 'hash_author']

        self.df_author_info = self.df_author_info.merge(df_author_pagerank, on='hash_author', how='left')
        self.df_author_info.sort_values(by=['pagerank'], ascending=False, inplace=True)
        print('Top authors by pagerank')
        print(self.df_author_info.head())

        del graph
        del df_temp
        del df_temp_scaled
        del df_author_pagerank
        gc.collect()

    def set_paper_pagerank(self):
        print('--- Set Paper Pagerank')

        print('Adding papers to graph')
        self.df_bibliographies['weight'] = 1
        graph = nx.from_pandas_edgelist(self.df_bibliographies, 'hash_id', 'referenced_hash_id', ['weight'],
                                        create_using=nx.DiGraph())

        print('Building paper pagerank')
        paper_pagerank = nx.pagerank(graph, tol=1e-9, alpha=0.8, dangling=None)

        df_paper_pagerank = pd.DataFrame.from_dict(paper_pagerank, orient='index')
        df_paper_pagerank['hash_id'] = df_paper_pagerank.index
        df_paper_pagerank.columns = ['pagerank', 'hash_id']

        self.df_paper_info = self.df_paper_info.merge(df_paper_pagerank, on='hash_id', how='left')
        self.df_paper_info.sort_values(by='pagerank', ascending=False)
        print('Top papers by pagerank')
        print(self.df_paper_info.head())

        del graph
        del df_paper_pagerank
        gc.collect()

    def consolidate_metrics(self):
        print('--- Consolidate Metrics')

        self.df_metrics = self.df_citations.merge(self.df_author_info, on='hash_author',
                                                                      how='left').drop(
            columns=['author_y', 'paper_count_x', 'paper_count_y']).rename(
            columns={'author_x': 'author', 'hash_author_x': 'hash_author', 'pagerank': 'author_pagerank',
                     'paper_count': 'author_paper_count'})

        self.df_metrics = self.df_metrics.merge(self.df_paper_info, on='paper_id', how='left').drop(
            columns=['title_y', 'pubmed_id_y', 'paper_citation_count_y']).rename(
            columns={'title_x': 'title', 'pubmed_id_x': 'pubmed_id', 'paper_citation_count_x': 'paper_citation_count',
                     'pagerank': 'paper_pagerank'})

        self.df_metrics['author_pagerank_aggr'] = self.df_metrics.groupby(['paper_id'])[
            'author_pagerank'].transform(np.mean)

        # this did group by paper id
        self.df_metrics['author_citation_ratio_sum'] = \
            self.df_metrics.groupby(['hash_author'])['author_citation_ratio'].transform(sum)

        self.df_metrics = self.df_metrics.drop(
            columns=['hash_author', 'author', 'author_citation_ratio', 'author_pagerank']).reset_index()

        self.df_metrics = self.df_metrics.drop_duplicates(
            subset=['paper_id']).reset_index()

        self.df_metrics.sort_values(by='paper_pagerank', ascending=False)

    def calculate_paper_metrics(self, row):
        weights_mf1 = [1]
        weights_mf2 = [1]
        weights_mf3 = [1, 1]
        weights_mf4 = [1, 1, 1]

        score_mf1 = 0.00
        score_mf1 = 0.00
        score_mf1 = 0.00
        score_mf1 = 0.00

        year_weight = 0
        # Year (Recency)
        if row['publish_year'] > self.current_year:
            publish_year = self.current_year
        else:
            publish_year = row['publish_year']
        try:
            year_weight = 1 / math.sqrt((self.current_year - publish_year) + 1)
        except:
            print(row['publish_year'])
        if year_weight < 0:
            year_weight = 0

        score_mf1 = weights_mf1[0] * row['paper_citation_count_norm']
        score_mf2 = weights_mf2[0] * row['author_citation_ratio_sum']

        # Only paper pagerank and author pagerank sum
        score_mf3 = ((weights_mf3[0] * row['paper_pagerank']) + (weights_mf3[1] * row['author_pagerank_aggr']))

        # Added year relevancy
        score_mf4 = ((weights_mf4[0] * row['paper_pagerank']) + (weights_mf4[1] * row['author_pagerank_aggr']) +
                     (weights_mf4[2] * year_weight))

        return pd.Series([score_mf1, score_mf2, score_mf3, score_mf4])

    def calculate_composite_metrics(self):
        print('--- Calculate Composite Metrics')

        self.df_metrics['paper_citation_count_norm'] = \
            self.df_metrics['paper_citation_count']

        cols_to_normalize = ['paper_citation_count_norm', 'paper_pagerank', 'author_pagerank_aggr',
                             'author_citation_ratio_sum']

        x = self.df_metrics[cols_to_normalize].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=cols_to_normalize, index=self.df_metrics.index)
        self.df_metrics[cols_to_normalize] = df_temp

        self.df_metrics['score_mf1'] = 0.00
        self.df_metrics['score_mf2'] = 0.00
        self.df_metrics['score_mf3'] = 0.00
        self.df_metrics['score_mf4'] = 0.00

        self.df_metrics[['score_mf1', 'score_mf2', 'score_mf3', 'score_mf4']] = \
            self.df_metrics.apply(lambda row: self.calculate_paper_metrics(row), axis=1)

        cols_to_normalize = ['score_mf1', 'score_mf2', 'score_mf3', 'score_mf4']
        x = self.df_metrics[cols_to_normalize].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=cols_to_normalize, index=self.df_metrics.index)
        self.df_metrics[cols_to_normalize] = df_temp

        self.df_metrics = pd.merge(self.df_metrics, self.df[['hash_id', 'journal', 'authors']],
                                   on='hash_id', how='left')

        del self.df
        gc.collect()

        df_paper_bibliographies_count = self.df_bibliographies.groupby('paper_id')[
            'referenced_hash_id'].count().reset_index().rename(
            columns={'referenced_hash_id': 'referenced_papers_count'})

        df_paper_bibliographies_count.head()

        self.df_metrics = self.df_metrics.merge(df_paper_bibliographies_count, on='paper_id', how='left').fillna(0)

        self.df_metrics = self.df_metrics[
            ['hash_id', 'paper_id', 'title', 'pubmed_id', 'publish_year', 'publish_time', 'author_count',
             'referenced_papers_count', 'paper_citation_count', 'paper_citation_count_norm', 'paper_pagerank',
             'author_pagerank_aggr', 'author_citation_ratio_sum', 'score_mf1', 'score_mf2', 'score_mf3', 'score_mf4']]

        self.df_metrics.sort_values(by=['score_mf1'], ascending=False, inplace=True)
        print('Top papers by score mf1')
        print(self.df_metrics.head())

        self.df_metrics.sort_values(by=['score_mf2'], ascending=False, inplace=True)
        print('Top papers by score mf2')
        print(self.df_metrics.head())

        self.df_metrics.sort_values(by=['score_mf3'], ascending=False, inplace=True)
        print('Top papers by score mf3')
        print(self.df_metrics.head())

        self.df_metrics.sort_values(by=['score_mf4'], ascending=False, inplace=True)
        print('Top papers by score mf4')
        print(self.df_metrics.head())

    def collect_metrics_pre(self):
        self.articles_in = len(self.df)

    def log_metrics_post(self):
        self.run.log('# Articles In', self.articles_in)
        self.run.log('# Articles Out', len(self.df_metrics))

        # Log scoring approach distributions
        fig_name = 'Distribution Paper Citation Count'
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.distplot(self.df_metrics['paper_citation_count_norm'], color='red').set_title(fig_name)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close(fig)

        fig_name = 'Distribution Paper PageRank'
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.distplot(self.df_metrics['paper_pagerank'], color='red').set_title(fig_name)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close(fig)

        fig_name = 'Distribution Paper Author PageRank Aggr'
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.distplot(self.df_metrics['author_pagerank_aggr'], color='red').set_title(fig_name)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close(fig)

        fig_name = 'Distribution Author Citation Ratio Sum'
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.distplot(self.df_metrics['author_citation_ratio_sum'], color='red').set_title(fig_name)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close(fig)

        fig_name = 'Distribution Metric Function 1'
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.distplot(self.df_metrics['score_mf1'], color='red').set_title(fig_name)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close(fig)

        fig_name = 'Distribution Metric Function 2'
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.distplot(self.df_metrics['score_mf2'], color='red').set_title(fig_name)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close(fig)

        fig_name = 'Distribution Metric Function 3'
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.distplot(self.df_metrics['score_mf3'], color='red').set_title(fig_name)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close(fig)

        fig_name = 'Distribution Metric Function 4'
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.distplot(self.df_metrics['score_mf4'], color='red').set_title(fig_name)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close(fig)

    def output_datasets(self):
        print('--- Output Datasets')
        if not (self.args.output_author_article_metrics is None):
            os.makedirs(self.args.output_author_article_metrics, exist_ok=True)
            path = self.args.output_author_article_metrics + '/processed.csv'
            self.df_metrics.to_csv(path, index=False)
            print('Output author article metrics created: {}'.format(path))
            print('Column definition of output author article metrics')
            print(self.df_metrics.columns)

        if not (self.args.output_author_info is None):
            os.makedirs(self.args.output_author_info, exist_ok=True)
            path = self.args.output_author_info + '/processed.csv'
            self.df_author_info.to_csv(path, index=False)
            print('Output author info created: {}'.format(path))
            print('Column definition of output author info')
            print(self.df_author_info.columns)

        if not (self.args.output_authors_by_paper is None):
            os.makedirs(self.args.output_authors_by_paper, exist_ok=True)
            path = self.args.output_authors_by_paper + '/processed.csv'
            self.df_authors_by_paper.to_csv(path, index=False)
            print('Output authors by paper created: {}'.format(path))
            print('Column definition of output author info')
            print(self.df_authors_by_paper.columns)

    def offline_save_fig(self, name):
        if 'OfflineRun' in self.run._identity:
            full_path = 'plots/' + name + '.png'
            plt.savefig(full_path, dpi=300, format='png')


if __name__ == '__main__':
    print('--- Author Article Metrics Started')
    author_article_metrics = AuthorArticleMetrics()
    print('--- Author Article Metrics Completed')
