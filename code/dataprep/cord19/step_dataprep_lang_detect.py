import argparse
import os
from azureml.core import Run
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scispacy
import spacy
import en_core_sci_lg
from spacy_langdetect import LanguageDetector


class LangDetect:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.df = None
        self.articles_in = 0
        self.articles_non_en = 0
        self.nlp = None

        self.get_runtime_arguments()

        self.load_dataset()
        self.set_nlp_model()

        self.collect_metrics_pre()
        self.lang_detect()
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
            help='Output extract data'
        )
        parser.add_argument(
            '--max_doc_length',
            type=int,
            help='Max doc length'
        )

        self.args = parser.parse_args()

        print('Input: {}'.format(self.args.input))
        print('Output: {}'.format(self.args.output))
        print('Max doc length: {}'.format(self.args.max_doc_length))

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

    def set_nlp_model(self):
        print('--- Set NLP Model')
        self.nlp = en_core_sci_lg.load(disable=['tagger', 'ner'])
        self.nlp.max_length = 1000000
        self.nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    def detect_article_lang(self):
        print('--- Detect Article Language')
        self.df['text_language'] = self.df.body_text.apply(lambda x: self.nlp(str(x[:int(self.args.max_doc_length)]))._.language['language'])
        articles_by_lang = self.df['text_language'].value_counts()
        print(articles_by_lang)

        self.articles_non_en = self.df.loc[self.df[self.df.text_language != 'en'].index].shape
        print('Number of non-english articles: {}'.format(str(self.articles_non_en)))

        df_temp = pd.DataFrame({'language': articles_by_lang.index, 'Count': articles_by_lang.values})
        df_temp.sort_values(by=['Count'], inplace=True)

        fig_name = 'Articles by Lang'
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.barplot(x='language', y='Count', data=df_temp, palette='husl')
        plt.xlabel('Language', fontsize=24)
        plt.ylabel('Articles', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        self.run.log_image(fig_name, plot=fig)
        self.offline_save_fig(fig_name)
        plt.close()

    def offline_save_fig(self, name):
        if 'OfflineRun' in self.run._identity:
            full_path = 'plots/' + name + '.png'
            plt.savefig(full_path, dpi=300, format='png')

    def drop_non_english_articles(self):
        print('--- Drop Non-English Articles')
        self.df = self.df.drop(self.df[self.df.text_language != 'en'].index)
        print('Number remaining English language articles: {}'.format(str(self.df.shape[0])))

    def lang_detect(self):
        self.detect_article_lang()
        self.drop_non_english_articles()

    def collect_metrics_pre(self):
        self.articles_in = len(self.df)

    def log_metrics_post(self):
        self.run.log('# Articles In', self.articles_in)
        self.run.log('# Articles Non-En', self.articles_non_en)
        self.run.log('# Articles Out', len(self.df))

    def output_dataset(self):
        print('--- Output Dataset')
        self.df.drop(columns=['text_language'], inplace=True)
        if not (self.args.output is None):
            os.makedirs(self.args.output, exist_ok=True)
            path = self.args.output + "/processed.csv"
            self.df.to_csv(path, index=False)
            print('Output created: {}'.format(path))
            print('Column definition of output')
            print(self.df.columns)


if __name__ == "__main__":
    print('--- Language Detection Started')
    lang_detect = LangDetect()
    print('--- Language Detection Completed')