import argparse
import os
from azureml.core import Run
import pandas as pd
import numpy as np
from collections import OrderedDict


class ConsolidateOutput:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.article_incoming_sources = OrderedDict()

        self.article_incoming_sources['input_hash_id'] = ['hash_id', 'paper_id', 'title', 'source', 'doi', 'pubmed_id',
                                                          'journal', 'abstract', 'publish_time', 'url', 'bibliography',
                                                          'authors']

        self.article_incoming_sources['input_stopwords_lemma'] = ['hash_id', 'body_text', 'processed_body_text']

        self.article_incoming_sources['input_feature_engineering'] = ['hash_id', 'publish_year', 'is_coronavirus']

        self.article_incoming_sources['input_study_design'] = ['hash_id', 'study_design']

        self.df_incoming = None
        self.df_consolidated_article = pd.DataFrame()

        self.get_runtime_arguments()

        self.collect_metrics_pre()
        self.consolidate_article_data()
        self.log_metrics_post()

        self.output_dataset()

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input_hash_index',
            type=str,
            help='Input hash index'
        )
        parser.add_argument(
            '--input_stopwords_lemma',
            type=str,
            help='Input stopwords lemmma'
        )
        parser.add_argument(
            '--input_topic_modelling',
            type=str,
            help='Input topic modelling'
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
            '--input_trials',
            type=str,
            help='Input trials'
        )
        parser.add_argument(
            '--output',
            type=str,
            help=' Output extract data'
        )

        self.args = parser.parse_args()

        print('Input Hash Index: {}'.format(self.args.input_hash_index))
        print('Input Stopwords Lemma: {}'.format(self.args.input_stopwords_lemma))
        print('Input Topic Modelling: {}'.format(self.args.input_topic_modelling))
        print('Input Feature Engineering: {}'.format(self.args.input_feature_engineering))
        print('Input Author Info: {}'.format(self.args.input_author_info))
        print('Input Authors By Paper: {}'.format(self.args.input_authors_by_paper))
        print('Input Trials: {}'.format(self.args.input_trials))
        print('Output: {}'.format(self.args.output))

    def consolidate_article_data(self):
        for k, v in self.article_incoming_sources.items():
            print('Processing {}'.format(k))
            path = vars(self.args).get(k) + '/processed.csv'
            self.df_incoming = pd.read_csv(path)
            self.df_incoming_stats()
            if len(self.df_consolidated_article) == 0:
                self.df_consolidated_article = self.df_incoming[v]
            else:
                self.df_consolidated_article = self.df_consolidated_article.merge(self.df_incoming[v], on='hash_id')
            self.df_consolidated_article_stats()


        #self.add_stopwords_lemma()
        #self.add_topic_modelling()
        #self.add_study_methods()
        #self.add_feature_engineering()
        #self.add_author_info()
        #self.add_authors_by_paper()
        #self.add_trials()

    def add_stopwords_lemma(self):
        print('--- Adding Stopwords Lemma Data')

        # Adds hash_id, paper_id, body_text, processed_body_text
        try:
            path = self.args.input_stopwords_lemma + '/processed.csv'
            self.df_incoming = pd.read_csv(path)
            self.df_incoming_stats()
            self.df_consolidated_article = self.df_incoming.copy()
            self.df_consolidated_article_stats()
        except:
            print('Error adding stopwords lemma')

    def add_topic_modelling(self):
        print('--- Adding Topic Modelling Data')

        try:
            path = self.args.input_topic_modelling + '/processed.csv'
            df = pd.read_csv(path)
            print('Raw Topic Modelling Specifications')
            print(df.head())
            print(df.columns)
            print(df.shape)
        except:
            print('Error adding topic modelling')

    def add_feature_engineering(self):
        print('--- Adding Feature Engineering Data')

        # Adds publish_time, publish_year, journal, url, pubmed_id, abstract, is_coronavirus
        try:
            path = self.args.input_feature_engineering + '/processed.csv'
            self.df_incoming = pd.read_csv(path)
            self.df_incoming_stats()
            print('Raw Feature Engineering Specifications')
            print(df.head())
            print(df.columns)
            print(df.shape)
            df = df[['hash_id', 'publish_time', 'publish_year', 'journal', 'url', 'pubmed_id', 'abstract', 'is_coronavirus']]
            self.df = self.df.merge(df, on='hash_id')
            self.df_stats()

        except:
            print('Error adding feature engineering')

    def add_author_info(self):
        print('--- Adding Author Info Data')

        try:
            path = self.args.input_author_info + '/processed.csv'
            df = pd.read_csv(path)

            print('Raw Author Info Specifications')
            print(df.head())
            print(df.columns)
            print(df.shape)
        except:
            print('Error adding author info')

    def add_authors_by_paper(self):
        print('--- Adding Authors By Paper Data')

        try:
            path = self.args.input_authors_by_paper + '/processed.csv'
            df = pd.read_csv(path)

            print('Raw Authors By Paper Specifications')
            print(df.head())
            print(df.columns)
            print(df.shape)
        except:
            print('Error adding authors by paper')

    def add_trials(self):
        print('--- Load Trials Data')

        try:
            path = self.args.input_trials + '/processed.csv'
            df = pd.read_csv(path)
            print('Raw Trials Specifications')
            print(df.head())
            print(df.columns)
            print(df.shape)
        except:
            print('Error adding trials')

    def df_incoming_stats(self):
        print('Consolidated Article DataFrame Stats')
        print(self.df_incoming.head())
        print(self.df_incoming.columns)
        print(self.df_incoming.shape)

    def df_consolidated_article_stats(self):
        print('Consolidated Article DataFrame Stats')
        print(self.df_consolidated_article.head())
        print(self.df_consolidated_article.columns)
        print(self.df_consolidated_article.shape)


    def collect_metrics_pre(self):
        pass
        #self.articles_in = len(self.df)

    def log_metrics_post(self):
        pass
        #self.run.log('# Articles In', self.articles_in)
        #self.run.log('# Articles Non-En', self.articles_non_en)
        #self.run.log('# Articles Out', len(self.df))

    def output_dataset(self):
        print('--- Output Dataset')
        if not (self.args.output is None):
            os.makedirs(self.args.output, exist_ok=True)
            path = self.args.output + "/processed.csv"
            print('Output created: {}'.format(path))
            self.df_consolidated_article.to_csv(path, index=False)


if __name__ == "__main__":
    print('--- Consolidate Output Started')
    consolidate_output = ConsolidateOutput()
    print('--- Consolidate Output Completed')
