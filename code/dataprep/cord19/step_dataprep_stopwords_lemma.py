import argparse
import os
from azureml.core import Run
import numpy as np
import pandas as pd
from tqdm import tqdm
import scispacy
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string


class StopwordsLemma:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.df = None
        self.articles_in = 0
        self.nlp = None
        self.punctuations = string.punctuation
        self.stopwords = None

        self.get_runtime_arguments()
        self.load_dataset()

        self.collect_metrics_pre()
        self.set_nlp_model()
        self.set_stopwords()
        self.tokenize()
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
            'url': str,
            'hash_id': str})

        print('Raw Input Specifications')
        print(self.df.head())
        print(self.df.columns)
        print(self.df.shape)

        print('Input Following Column Subset')
        self.df = self.df[['hash_id', 'paper_id', 'title', 'abstract', 'publish_time', 'subset_source']]
        print(self.df.head())
        print(self.df.columns)
        print(self.df.shape)

    def set_nlp_model(self):
        print('--- Set NLP Model')
        self.nlp = spacy.load('en_core_sci_lg', disable=["tagger", "ner"])
        self.nlp.max_length = 7000000

    def set_stopwords(self):
        print('--- Set Stopwords')
        self.stopwords = list(STOP_WORDS)

        custom_stop_words = [
                 'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
                 'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
                 'al.', 'Elsevier', 'PMC', 'CZI', 'www', 'amsmath', 'amsbsy', 'estimate', 'parameter', 'network',
                 'display', 'version', 'approach', 'end', 'begin', 'minimal', 'amsfonts', 'amssymb', 'amsmath',
                 'documentclass', 'amsbsy', 'size', 'apply', 'usepackage', 'document', 'month', 'author/funder',
                 'doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'table',
                 'rights', 'reserved', 'permission', 'use', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
                 'al.', 'Elsevier', 'PMC', 'CZI',
                 '-PRON-', 'usually',
            ]
        for w in custom_stop_words:
            if w not in self.stopwords:
                self.stopwords.append(w)

    def spacy_tokenizer(self, sentence):
        sentence_tokens = self.nlp(sentence)
        sentence_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in
                           sentence_tokens]
        sentence_tokens = [word for word in sentence_tokens if word not in self.stopwords and word not in
                           self.punctuations]
        sentence_tokens = ' '.join([i for i in sentence_tokens])
        return sentence_tokens

    def tokenize(self):
        print('--- Applying Stopword Tokenization')
        self.df['title'] = self.df['title'].astype(str)
        self.df['abstract'] = self.df['abstract'].astype(str)  # Known to contain just float

        self.df['processed_title'] = self.df['title'].apply(self.spacy_tokenizer)
        self.df['processed_abstract'] = self.df['abstract'].apply(self.spacy_tokenizer)

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
    print('--- Stopwords Lemma Started')
    stopwords_lemma = StopwordsLemma()
    print('--- Stopwords Lemma Completed')
