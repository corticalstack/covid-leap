import joblib
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, models, CrossEncoder
import copy
import json
from urllib.parse import urlparse, parse_qs
from sqlalchemy import create_engine
from elasticsearch import Elasticsearch, helpers
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication
from sklearn.metrics.pairwise import cosine_similarity
import torch
from collections import OrderedDict
# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.


def init():
    print('Inference Init')
    global model
    global df
    global dataset
    global psql_host
    global psql_user
    global psql_pwd
    global dbname
    global es_conn
    global neural_model_embedding
    global cross_encoder

    psql_host = os.environ.get('AZ_RP_PSQL_HOST')
    psql_user = os.environ.get('AZ_RP_PSQL_USER')
    psql_pwd = os.environ.get('AZ_RP_PSQL_PWD')
    dbname = 'analytics-shared-repository'

    es_host = os.environ.get('AZ_RP_ES_HOST')
    es_user = os.environ.get('AZ_RP_ES_USER'                                  '')
    es_pwd = os.environ.get('AZ_RP_ES_PWD')

    es_conn = Elasticsearch([es_host], http_auth=(es_user, es_pwd), scheme="https", port=443, timeout=60)

    svc_key = 'UmOO0.xQaa_g_-2MgTwS_p72EN1EVJg6Tv'

    svc_pr = ServicePrincipalAuthentication(
        tenant_id='ee1a033c-2d44-4728-891d-4ef598619020',
        service_principal_id='bc9ed0dc-9d90-403e-b4bb-58a067749604',
        service_principal_password=svc_key)

    ws = Workspace(
        subscription_id="19518d47-0c8b-4829-a602-c5ced78deb3f",
        resource_group="aci-eur-frc-aa-ss-rg",
        workspace_name="aci-eur-frc-aa-ss-mlw",
        auth=svc_pr
    )

    neural_model_path = Model.get_model_path("c19gq_ance_msmarco_passage", None, ws)
    neural_model_embedding = SentenceTransformer(neural_model_path)
    if torch.cuda.is_available():
        neural_model_embedding = neural_model_embedding.to(torch.device('cuda'))

    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')


def get_complimentary_data(conn, hash_id, response):
    sql_string = "SELECT publish_year, journal, doi, url FROM pub_article " \
                 "WHERE hash_id in {}".format(tuple(hash_id))
    _df = pd.read_sql(sql_string, conn)
    for index, row in _df.iterrows():
        if row['url']:
            response[index]['Url'] = row['url']
        if row['doi']:
            response[index]['Doi'] = row['doi']
        if row['publish_year']:
            response[index]['Publish Year'] = row['publish_year']
        if row['journal']:
            response[index]['Journal'] = row['journal']
    return response


def get_trials_data(conn):
    #sql_string = "SELECT * from pub_trials"
    sql_string = "SELECT hash_id, string_agg(id, ', ') as trial_ids FROM pub_trials GROUP BY hash_id"
    _df = pd.read_sql(sql_string, conn)
    return _df



def get_filter_terms(**kwargs):
    terms = {}
    if 'virus_constraint' in kwargs:
        if kwargs['virus_constraint'] == 'Sars-CoV-2':
            terms['is_sars_cov2'] = ['true']
        elif kwargs['virus_constraint'] == 'Sars-CoV-2 Strict':
            terms['is_sars_cov2_title'] = ['true']
        elif kwargs['virus_constraint'] == 'Sars-CoV':
            terms['is_sars_cov'] = ['true']
        elif kwargs['virus_constraint'] == 'Sars-CoV Strict':
            terms['is_sars_cov_title'] = ['true']
        elif kwargs['virus_constraint'] == 'Mers':
            terms['is_mers'] = ['true']
        elif kwargs['virus_constraint'] == 'Mers Strict':
            terms['is_mers_title'] = ['true']
        elif kwargs['virus_constraint'] == 'Coronavirus':
            terms['is_coronavirus'] = ['true']
        elif kwargs['virus_constraint'] == 'Coronavirus Strict':
            terms['is_coronavirus_title'] = ['true']
    return terms


def get_response_fields():
    return ['hash_id',
            'title',
            'publish_year',
            'journal',
            'topic_id',
            'doi',
            'pubmed_id',
            'text',
            'is_coronavirus',
            'is_coronavirus_title',
            'is_sars_cov2',
            'is_sars_cov2_title',
            'is_sars_cov',
            'is_sars_cov_title',
            'is_mers',
            'is_mers_title',
            'author_count',
            'paper_citation_count',
            'paper_pagerank',
            'score_mf1',
            'score_mf2',
            'score_mf3',
            'score_mf4',
            'text_processed_vector']


def get_body(**kwargs):
    fields = get_response_fields()

    terms = {}
    if 'virus_constraint' in kwargs:
        terms = get_filter_terms(virus_constraint=kwargs['virus_constraint'])
    print('Terms is:', terms)

    if 'q' in kwargs:
        match = {
                    'multi_match': {
                        'query': kwargs['q'],
                        'fields': ['title',
                                   'text']
                    }
                }
    elif 'qv' in kwargs:
        match = {
                    'script_score': {
                        'query': {
                            'match_all': {}
                        },
                        'script': {
                            'source': 'cosineSimilarity(params.queryVector, doc["text_processed_vector"]) + 1.0',
                            'params': {
                                'queryVector': kwargs['qv']
                            }
                        }
                    }
                }

    body = {
        'query': {
            'bool': {
                'must': [
                    match,
                    {
                        'terms': terms
                    },
                    {
                        'range': {
                            'publish_year': {
                                'gte': kwargs['q_year_from'],
                                'lte': kwargs['q_year_to']
                            }
                        }
                    }
                ]
            }
        },
        'fields': fields,
        '_source': False
    }

    if not terms:
        del body['query']['bool']['must'][1]

    return body


def run(data):
    response = []
    entry = OrderedDict({
        'hash_id': '',
        'title': '',
        'score': 0,
        'cross_encode_score': 0,
        'rerank_score': 0,
        'publish_year': 0,
        'journal': '',
        'topic_id': 0,
        'doi': '',
        'pubmed_id': '',
        'url': '',
        'trials': '',
        'is_coronavirus': '',
        'is_coronavirus_title': '',
        'is_sars_cov2': '',
        'is_sars_cov2_title': '',
        'is_sars_cov': '',
        'is_sars_cov_title': '',
        'is_mers': '',
        'is_mers_title': '',
        'author_count': 0,
        'paper_citation_count': 0,
        'paper_pagerank': 0,
        'score_mf1': 0,
        'score_mf2': 0,
        'score_mf3': 0,
        'score_mf4': 0,
        'text': '',
        'text_processed_vector': []
        })

    o = urlparse(data)
    query = parse_qs(o.path)
    query_search_phrase = query['search_phrase'][0]
    query_search_type = query['search_type'][0]
    query_year_from = query['year_from'][0]
    query_year_to = query['year_to'][0]
    query_virus_constraint = query['virus_constraint'][0]
    query_search_size = int(query['search_size'][0])
    query_rerank_depth = int(query['rerank_depth'][0])

    print('Query is:', query_search_phrase)
    print('Search type:', query_search_type)
    print('Query year from:', query_year_from)
    print('Query year to:', query_year_to)
    print('Virus constraint:', query_virus_constraint)
    print('Search size:', query_search_size)
    print('Rerank depth:', query_rerank_depth)

    db_conn_path = 'postgresql://' + psql_user + ':' + psql_pwd + '@' + psql_host + ':5432/' + dbname
    sqlalchemy_engine = create_engine(db_conn_path, connect_args={'sslmode': 'require'})
    conn = sqlalchemy_engine.connect()

    query_embedding = neural_model_embedding.encode(query_search_phrase)

    # BM25 lexical
    if query_search_type == 'BM25' or query_search_type == 'BM25 + Semantic Rerank':
        body_text = get_body(q=query_search_phrase, virus_constraint=query_virus_constraint,
                             q_year_from=query_year_from, q_year_to=query_year_to)

    # Dense (semantic) search
    if query_search_type == 'Semantic' or query_search_type == 'Semantic Cross-Encoder':
        body_text = get_body(qv=query_embedding, virus_constraint=query_virus_constraint, q_year_from=query_year_from,
                             q_year_to=query_year_to)

    #if query_search_type == 'BM25 + Semantic Rerank' or query_search_type == 'Semantic Cross-Encoder':
    #    size = query_search_depth
    #else:
    #    size = query_number_results

    results = es_conn.search(index='pub_text', size=query_search_size, body=body_text)

    print('Number of results', str(len(results['hits']['hits'])))

    if query_search_type == 'Semantic Cross-Encoder':
        cross_inp = [[query_search_phrase, (hit['fields']['title'][0] + ' ' + hit['fields']['text'][0])] for hit in
                     results['hits']['hits'][:query_rerank_depth]]
        cross_scores = cross_encoder.predict(cross_inp)
        for idx in range(len(cross_scores)):
            results['hits']['hits'][idx]['cross_encode_score'] = cross_scores[idx]

    for hit in results['hits']['hits'][0:query_search_size]:
        new_entry = copy.deepcopy(entry)
        new_entry['hash_id'] = hit['fields']['hash_id'][0]
        new_entry['title'] = hit['fields']['title'][0]
        new_entry['score'] = hit['_score']

        if 'publish_year' in hit['fields']:
            new_entry['publish_year'] = hit['fields']['publish_year'][0]

        if 'journal' in hit['fields']:
            new_entry['journal'] = hit['fields']['journal'][0]

        new_entry['topic_id'] = hit['fields']['topic_id'][0]

        if 'doi' in hit['fields']:
            new_entry['doi'] = hit['fields']['doi'][0]

        if 'pubmed_id' in hit['fields']:
            new_entry['pubmed_id'] = hit['fields']['pubmed_id'][0]

        if 'url' in hit['fields']:
            new_entry['url'] = hit['fields']['url'][0]

        new_entry['author_count'] = hit['fields']['author_count'][0]
        new_entry['paper_citation_count'] = hit['fields']['paper_citation_count'][0]
        new_entry['paper_pagerank'] = hit['fields']['paper_pagerank'][0]
        new_entry['score_mf1'] = hit['fields']['score_mf1'][0]
        new_entry['score_mf2'] = hit['fields']['score_mf2'][0]
        new_entry['score_mf3'] = hit['fields']['score_mf3'][0]
        new_entry['score_mf4'] = hit['fields']['score_mf4'][0]
        new_entry['text'] = hit['fields']['text'][0]
        new_entry['text_processed_vector'] = hit['fields']['text_processed_vector']
        try:
            new_entry['cross_encode_score'] = str(hit['cross_encode_score'])
        except KeyError:
            pass
        response.append(new_entry)

    # BM25 + dense reranker
    if query_search_type == 'BM25 + Semantic Rerank':
        _df_reranked = pd.DataFrame(response[:query_rerank_depth], columns=entry)
        _df_reranked['q_vector'] = _df_reranked.apply(lambda x: query_embedding, axis=1)
        q_vector = np.array(list(_df_reranked['q_vector']))
        text_vector = np.array(list(_df_reranked['text_processed_vector']))
        _df_reranked['rerank_score'] = np.diag(cosine_similarity(q_vector, text_vector))
        _df_reranked.sort_values(by=['rerank_score'], ascending=False, inplace=True)
        _df_reranked.drop(columns=['q_vector'], inplace=True, axis=1)
        response = _df_reranked.to_dict(into=OrderedDict, orient='records')

    _df_trials = get_trials_data(conn)
    _df_trials['Trials'] = _df_trials[['hash_id', 'trial_ids']].groupby(['hash_id'])['trial_ids'].transform(lambda x: ', '.join(x))
    #_df_trials.drop(['id'], axis=1, inplace=True)
    _df_trials.drop_duplicates(subset='hash_id', keep='first', inplace=True)

    for idx, row in enumerate(response):
        if idx < 5:
            print(row)
        test = _df_trials.loc[_df_trials['hash_id'] == row['hash_id'], 'trial_ids']
        if len(test) > 0:
            row['trial_ids'] = test.to_string(header=False, index=False)
        else:
            row['trial_ids'] = ''

        print(test)
        #row['triaL_ids'] = _df_trials.loc[_df_trials['hash_id'] == row['hash_id'], 'trial_ids']
        row.pop('text_processed_vector', None)



    entries_as_json = json.dumps([*response])
    data = "{\"results\": " + entries_as_json + "}"

    return data

