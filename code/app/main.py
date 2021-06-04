import os
import datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_metrics import metric, metric_row
import streamlit_session_state as sss
import requests
import ast
import json
from collections import OrderedDict

from sqlalchemy import create_engine, exc
import sqlalchemy
import time


psql_host = os.environ.get('AZ_RP_PSQL_HOST')
psql_user = os.environ.get('AZ_RP_PSQL_USER')
psql_pwd = os.environ.get('AZ_RP_PSQL_PWD')
dbname = 'analytics-shared-repository'
db_conn_path = 'postgresql://' + psql_user + ':' + psql_pwd + '@' + psql_host + ':5432/' + dbname

# First index is platform, second index gives colour for the platform
platform_map = [['Protein Subunit', 'Viral Vector: Non-Replicating', 'SARS-CoV-2 Virus: Inactivated',
                 'Nucleic Acid: DNA', 'Nucleic Acid: RNA', 'SARS-CoV-2 Virus: Live Attenuated',
                 'Viral Vector: Non-Replicating + APC', 'Viral Vector: Replicating',
                 'Viral Vector: Replicating + APC', 'Virus Like Particle'],
                ['darkturquoise', 'thistle', 'lightslategray', 'darkgoldenrod', 'darkslategray', 'lightskyblue',
                 'lightgrey', 'tomato', 'bisque', 'khaki']]

platform_legend = []
for p in platform_map[0]:
    l_item = {'field': p, 'type': "quantitative"}
    platform_legend.append(l_item)


sql_string_read_trials_candidates = 'SELECT * from trials_candidates'
sql_string_read_trials_clinical_stage = 'SELECT * from trials_clinical_stage'


@st.cache(allow_output_mutation=True)
def get_sql_engine(path):
    """Put the connection in cache to reuse if path does not change."""
    sqlalchemy_engine = create_engine(path, connect_args={'sslmode': 'require'})
    return sqlalchemy_engine


@st.cache(hash_funcs={sqlalchemy.engine.base.Engine: id})
def get_sql_data(engine, sql_string):
    """If the Connection.id is the same as before, use the cached dataframe"""
    df = pd.read_sql(sql_string, engine.connect())
    return df


@st.cache(allow_output_mutation=True)
def search_articles(search_phrase, search_type, year_range, virus_constraint, search_size, rerank_depth):
    url_inference_url = 'http://localhost:6789/score'

    params = dict(search_phrase=search_phrase,
                  search_type=search_type,
                  year_from=year_range[0],
                  year_to=year_range[1],
                  virus_constraint=virus_constraint,
                  search_size=search_size,
                  rerank_depth=rerank_depth
                  )

    headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
    response = requests.post(url_inference_url, data=params, headers=headers)
    
    resp_as_dict = ast.literal_eval(json.loads(response.text))
    df = pd.DataFrame([i for i in resp_as_dict['results']])

    df.drop(columns=['hash_id', 'is_coronavirus', 'is_coronavirus_title', 'is_sars_cov2', 'is_sars_cov2_title',
                     'is_sars_cov', 'is_sars_cov_title', 'is_mers', 'is_mers_title', 'url', 'trials'], inplace=True)

    if search_type == 'BM25 + Semantic Rerank':
        df['Relevancy'] = df['rerank_score']
    elif search_type == 'Semantic Cross-Encoder':
        df['Relevancy'] = df['cross_encode_score']
    else:
        df['Relevancy'] = df['score']

    df['Relevancy'] = pd.to_numeric(df['Relevancy'])

    df.rename(columns={'title': 'Title', 'publish_year': 'Published Year', 'journal': 'Journal', 'topic_id': 'Topic',
                       'doi': 'DOI', 'pubmed_id': 'PubMed ID', 'author_count': 'Author Count', 'paper_citation_count':
                       'Paper Citation Count', 'paper_pagerank': 'Paper PageRank', 'text': 'Paragraph',
                       'trial_ids': 'Trials'}, inplace=True)

    df['Topic'] = df['Topic'].astype(str)
    df['Topic'].replace({'0': 'Clinical', '1': 'Virology', '2': 'Public Health', '3': 'Genomics', '4': 'Health Care'},
                        inplace=True)

    return df


def set_score_by_metric(df, search_type, applied_metric):

    if len(df) == 0:
        return

    if search_type == 'BM25 + Semantic Rerank':
        df['Relevancy'] = df['rerank_score']
    elif search_type == 'Semantic Cross-Encoder':
        df['Relevancy'] = df['cross_encode_score']
    else:
        df['Relevancy'] = df['score']

    df['Relevancy'] = pd.to_numeric(df['Relevancy'])

    if applied_metric == 'Citation Count':
        df['Relevancy'] = df['Relevancy'] * (df['score_mf1'] * 2)
    elif applied_metric == 'Author Citation Ratio':
        df['Relevancy'] = df['Relevancy'] * (df['score_mf2'] * 4)
    elif applied_metric == 'PageRank':
        df['Relevancy'] = df['Relevancy'] * (df['score_mf3'] * 10)
    elif applied_metric == 'Paper & Recency':
        df['Relevancy'] = df['Relevancy'] * (df['score_mf4'] * 0.7)

    df.sort_values(by=['Relevancy'], ascending=False, inplace=True)


@st.cache
def get_spec_overview_bar_chart(p_map):
    spec = {"width": "container",
            "height": {"step": 40},
            'encoding': {
                'x': {'field': 'Id',
                      'type': 'quantitative',
                      'axis': {'tickCount': 5, 'title': "", 'titleFontSize': 20, 'labelFontSize': 18}},
                'y': {'field': 'Platform',
                      'type': 'nominal',
                      'sort': "-x",
                      'axis': {'title': "", 'titleFontSize': 20, 'labelFontSize': 18, 'labelPadding': 10,
                               'labelLimit': 1000}},
                "color": {
                    "field": "Platform",
                    'legend': False,
                    "type": "nominal",
                    "scale": {
                        "domain": p_map[0],
                        "range": p_map[1]}},
                "tooltip": [
                    {"field": "Platform", "type": "nominal", "title": "Vaccine Platform/Type"},
                    {"field": "Id", "type": "quantitative", "title": "# Candidates"}
                ],
            },
            "layer": [{
                "mark": "bar",
            }, {
                "mark": {
                    "type": "text",
                    "align": "right",
                    "xOffset": -5,
                    "fontSize": 18
                },
                "encoding": {
                    "text": {"field": "Id", "type": "quantitative"},
                    "color": {
                        "value": "white"}
                }
            }]
            }
    return spec


@st.cache
def get_spec_overview_bar_chart_pct(p_map):
    spec = {"width": "container",
            "height": {"step": 40},
            'encoding': {
                'x': {'field': 'Id',
                      'type': 'quantitative',
                      "aggregate": "sum",
                      'axis': {'tickCount': 5, 'title': "", 'titleFontSize': 20, 'labelFontSize': 18}},
                'y': {'field': 'Platform',
                      'type': 'nominal',
                      'sort': "-x",
                      'axis': {'title': "", 'titleFontSize': 20, 'labelFontSize': 18, 'labelPadding': 10,
                               'labelLimit': 1000}},
                "color": {
                    "field": "Platform",
                    'legend': False,
                    "type": "nominal",
                    "scale": {
                        "domain": p_map[0],
                        "range": p_map[1]}},
                "tooltip": [
                    {"field": "Platform", "type": "nominal", "title": "Vaccine Platform/Type"},
                    {"field": "Id", "type": "quantitative", "title": "% Candidates"}
                ],
            },
            "layer": [{
                "mark": "bar",
            }, {
                "mark": {
                    "type": "text",
                    "align": "right",
                    "xOffset": -5,
                    "fontSize": 18
                },
                "encoding": {
                    "text": {"field": "Id", "type": "quantitative"},
                    "color": {
                        "value": "white"}
                }
            }]
            }
    return spec


@st.cache
def get_spec_clinical_stacked_bar_h(p_map):
    spec = {"width": "container",
            "height": {"step": 60},
            "resolve": {"scale": {"color": "independent"}},
            "layer": [
                {"mark": "bar",
                 "encoding": {
                    "tooltip": [
                         {"field": "Platform", "type": "nominal", "title": "Vaccine Platform/Type"},
                         {"aggregate": "count", "field": "Id", "type": "quantitative", "title": "# Candidates"}
                    ],
                    "x": {"aggregate": "count", "field": "Id", "type": "quantitative", "stack": "zero",
                          'axis': {'title': "Candidates", 'titleFontSize': 20, 'tickCount': 5, 'labelFontSize': 18}},
                    "y": {"field": "Phase", "type": "nominal",
                          'axis': {'title': "Phase", 'titleFontSize': 20, 'labelFontSize': 18}},
                    "color": {"field": "Platform",
                              "type": "nominal",
                              "title": "Vaccine Platform/Type",
                              'legend': {
                                  'labelFontSize': 16,
                                  'labelLimit': 1000,
                                  'titleFontSize': 16},
                              "scale": {
                                  "domain": p_map[0],
                                  "range": p_map[1]}
                              },
                 }},
                 {"mark": {"type": "text", "dx": -15, "dy": 3, "fontSize": 18},
                  "encoding": {
                    "x": {"aggregate": "count", "field": "Id", "type": "quantitative", "stack": "zero"},
                    "y": {"field": "Phase", "type": "nominal"},
                    "color": {"field": "Platform", "type": "nominal", "scale": {"range": ["white"]}, "legend": None},
                    "text": {"aggregate": "count", "field": "Id", "type": "quantitative"}}
                 }
              ]
            }

    return spec


@st.cache
def get_spec_clinical_stacked_bar_h_pct(p_map):
    spec = {"width": "container",
            "height": {"step": 60},
            "resolve": {"scale": {"color": "independent"}},
            "layer": [
                {"mark": "bar",
                 "encoding": {
                    "tooltip": [
                         {"field": "Platform", "type": "nominal", "title": "Vaccine Platform/Type"},
                         {"aggregate": "sum", "field": "Id", "type": "quantitative", "title": "% Candidates"}
                    ],
                    "x": {"aggregate": "sum", "field": "Id", "type": "quantitative", "stack": "zero",
                          'axis': {'title': "Candidates", 'titleFontSize': 20, 'tickCount': 5, 'labelFontSize': 18}},
                    "y": {"field": "Phase", "type": "nominal",
                          'axis': {'title': "Phase", 'titleFontSize': 20, 'labelFontSize': 18}},
                    "color": {"field": "Platform",
                              "type": "nominal",
                              "title": "Vaccine Platform/Type",
                              'legend': {
                                  'labelFontSize': 16,
                                  'labelLimit': 1000,
                                  'titleFontSize': 16},
                              "scale": {
                                  "domain": p_map[0],
                                  "range": p_map[1]}
                              },
                 }},
                 {"mark": {"type": "text", "dx": -25, "dy": 3, "fontSize": 18},
                  "encoding": {
                    "x": {"aggregate": "sum", "field": "Id", "type": "quantitative", "stack": "zero"},
                    "y": {"field": "Phase", "type": "nominal"},
                    "color": {"field": "Platform", "type": "nominal", "scale": {"range": ["white"]}, "legend": None},
                    "text": {"aggregate": "sum", "field": "Id", "type": "quantitative"}}
                 }
              ]
            }

    return spec


@st.cache
def get_spec_clinical_multi_series_line_i(p_map):
    spec = {"height": 600,
            "resolve": {"scale": {"color": "independent"}},
            "encoding": {"x": {"field": "Publish Date",
                               "type": "temporal",
                               'axis': {'tickCount': 30,
                                        'title': "Publish Date",
                                        'titleFontSize': 20,
                                        'labelFontSize': 18}}},
            "layer": [
                {
                    "encoding": {
                        "color": {"field": "Platform",
                                  "type": "nominal",
                                  "scale": {
                                      "domain": p_map[0],
                                      "range": p_map[1]},
                                  'legend': {
                                      'symbolType': 'square',
                                      'labelFontSize': 16,
                                      'labelLimit': 1000,
                                      'titleFontSize': 16},
                                  },
                        "y": {"field": "Id",
                              "type": "quantitative",
                              'axis': {'tickCount': 10,
                                       'title': "# Candidates",
                                       'titleFontSize': 20,
                                       'labelFontSize': 18}},
                    },
                    "layer": [
                        {"mark": "line"},
                        {"transform": [{"filter": {"selection": "hover"}}], "mark": "point"}
                    ]
                },
                {
                    "transform": [{"pivot": "Platform", "value": "Id", "groupby": ["Publish Date"]}],
                    "mark": "rule",
                    "encoding": {
                        "opacity": {
                            "condition": {"value": 0.3, "selection": "hover"},
                            "value": 0
                        },
                        "tooltip": platform_legend
                    },
                    "selection": {
                        "hover": {
                            "type": "single",
                            "fields": ["Publish Date"],
                            "nearest": True,
                            "on": "mouseover",
                            "empty": "none",
                            "clear": "mouseout"
                        }
                    }
                }
            ]
            }

    return spec


def unique_publish_date(df):
    return sorted(df['Publish Date'].unique(), reverse=True)


def unique_journals(df):
    if len(df) > 0:
        return sorted(df['Journal'].unique())


def unique_topics(df):
    if len(df) > 0:
        return sorted(df['Topic'].unique())


def main():
    global df_trials_candidates
    global df_trials_clinical_stage
    global column_dict

    st.set_page_config(page_title='COVID-LEAP', page_icon='src/assets/img/cs-logo.png', layout='wide',
                       initial_sidebar_state='auto')

    state = sss._get_state()
    pages = {
        'Vaccine Overview': page_vaccine_overview,
        'Clinical Candidates': page_vaccine_clinical_candidates,
        'Clinical Candidates To Date': page_vaccine_clinical_candidates_to_date,
        'Preclinical Candidates': page_vaccine_preclinical_candidates,
        'Info': page_info,
        'Article Search': page_article_search
    }

    # Get trials data
    engine = get_sql_engine(db_conn_path)
    df_trials_candidates = get_sql_data(engine, sql_string_read_trials_candidates)
    df_trials_candidates.rename(columns={'id': 'Id',
                                         'publisher': 'Publisher',
                                         'publish_date': 'Publish Date',
                                         'evaluation_type': 'Evaluation Type',
                                         'disease': 'Disease',
                                         'manufacturer': 'Developer',
                                         'platform': 'Platform',
                                         'type': 'Type',
                                         'number_doses': 'Number Doses',
                                         'timing_doses': 'Timing Doses',
                                         'route_administration': 'Route of Administration',
                                         'shared_platform': 'Shared Platform',
                                         'phase': 'Phase'}, inplace=True)

    df_trials_clinical_stage = get_sql_data(engine, sql_string_read_trials_clinical_stage)
    df_trials_clinical_stage.rename(columns={'id': 'Id',
                                             'publisher': 'Publisher',
                                             'publish_date': 'Publish Date',
                                             'phase': 'Phase',
                                             'trial_id': 'Trial Id'}, inplace=True)

    page = st.sidebar.radio('Select your page', tuple(pages.keys()))
    pages[page](state)  # Display the selected page with the session state

    state.sync()  # Mandatory to avoid rollbacks with widgets, must be called at the end of app


def my_widget(key):
    st.subheader('Hello there!')
    clicked = st.button("Click me " + key)


def get_year_range(year_selection, custom_year_range):
    year_range = []
    current_year = datetime.datetime.now().year

    if year_selection == 'Custom':
        year_range.append(custom_year_range[0])
        year_range.append(custom_year_range[1])
    elif year_selection == 'Since 2021':
        year_range = [2021] * 2
    elif year_selection == 'Since 2020':
        year_range = [2020, current_year]
    else:
        year_range = [2019, current_year]
    return year_range


def page_article_search(state):
    metric_row(
        {
            'COVID-19 Research Cockpit': 'Article Search'
        }
    )
    df_search = pd.DataFrame()
    df_display = pd.DataFrame()
    search_time = 0

    if state.search_phrase:
        year_range = get_year_range(state.year_selection, state.custom_year_range)
        search_start = datetime.datetime.now()
        df_search = search_articles(state.search_phrase, state.filter_search_type, year_range,
                                    state.filter_virus_constraint, state.search_size,
                                    state.rerank_depth)
        search_complete = datetime.datetime.now()
        search_time = search_complete - search_start
        df_search['Published Year'] = pd.to_numeric(df_search['Published Year'])
        df_search.index = [""] * len(df_search)  # Hide pd index

    if state.applied_metric:
        set_score_by_metric(df_search, state.filter_search_type, state.applied_metric)

    df_display = df_search.copy()

    with st.sidebar:
        exp_search_options = st.beta_expander('Search Options', expanded=False)
        exp_results_options = st.beta_expander('Results Options', expanded=True)
        exp_tech_options = st.beta_expander('Technical Options', expanded=False)

        state.number_results = exp_results_options.slider('Number of Results', 10, 50, state.number_results, step=10)
        state.sort_by = exp_results_options.selectbox('Sort by',
                                                      ['Most relevant',
                                                       'Most citations',
                                                       'Title'])

        if state.search_phrase:
            if state.number_results:
                df_display = df_display[:state.number_results]
            fields_journal = unique_journals(df_display)
            fields_topic = unique_topics(df_display)
            state.filter_journal = exp_results_options.multiselect('Filter by journal', fields_journal)
            state.filter_topic = exp_results_options.multiselect('Filter by topic', fields_topic)

            if state.filter_journal:
                df_display = df_display[df_display['Journal'].isin(state.filter_journal)]

            if state.filter_topic:
                df_display = df_display[df_display['Topic'].isin(state.filter_topic)]

        state.year_selection = exp_search_options.radio('Filter by publication year',
                                                        ('Since 2021', 'Since 2020', 'Since 2019', 'Custom'))

        if state.year_selection == 'Custom':
            state.custom_year_range = exp_search_options.slider('Year range', 1950, 2021, (1950, 2021), 1)

        state.filter_virus_constraint = exp_search_options.selectbox('Virus Constraint Filter',
                                                                    ['None', 'Sars-CoV-2',
                                                                    'Sars-CoV-2 Strict',
                                                                    'Sars-CoV',
                                                                    'Sars-CoV Strict',
                                                                    'Mers',
                                                                    'Mers Strict',
                                                                    'Coronavirus',
                                                                    'Coronavirus Strict'])
        state.filter_search_type = exp_tech_options.selectbox('Search Type', ['BM25', 'Semantic',
                                                                              'BM25 + Semantic Rerank',
                                                                              'Semantic Cross-Encoder'])

        state.search_size = 1000
        state.search_size = exp_tech_options.slider('Search Size', 100, 1000, state.search_size, step=100)
        state.rerank_depth = 100
        state.rerank_depth = exp_tech_options.slider('Rerank Depth', 100, 200, state.rerank_depth, step=10)

        state.applied_metric = exp_tech_options.selectbox('Apply Paper Metric', ['None', 'Citation Count',
                                                                                 'Author Citation Ratio',
                                                                                 'PageRank', 'Paper & Recency'])
        state.hide_tech_columns = exp_tech_options.checkbox('Hide Technical Columns', value=True)
    state.search_phrase = st.text_input("Search articles for", "")

    if state.search_phrase and len(df_display) > 0:
        if state.sort_by == 'Most citations':
            df_display.sort_values(by=['Paper Citation Count'], ascending=False, inplace=True)
        elif state.sort_by == 'Title':
            df_display.sort_values(by=['Title'], ascending=True, inplace=True)
        else:
            df_display.sort_values(by=['Relevancy'], ascending=False, inplace=True)
        if state.hide_tech_columns:
            df_display.drop(['score', 'score_mf1', 'score_mf2', 'score_mf3', 'score_mf4', 'cross_encode_score',
                             'rerank_score', 'Paper PageRank'], axis=1, inplace=True)

        st.table(df_display)

    #st.write(search_time)

def page_vaccine_overview(state):
    fields_publish_date = unique_publish_date(df_trials_candidates)
    state.filter_publish_date = st.sidebar.selectbox('Select WHO Publication Date', fields_publish_date,
                                                     fields_publish_date.index(state.filter_publish_date)
                                                     if state.filter_publish_date else 0)

    df_trials_clinical = df_trials_candidates[(df_trials_candidates['Publish Date'] == state.filter_publish_date) &
                                              (df_trials_candidates['Evaluation Type'] == 'clinical')]
    df_trials_clinical_count = df_trials_clinical.groupby(['Platform']).count().reset_index()

    df_trials_preclinical = df_trials_candidates[(df_trials_candidates['Publish Date'] == state.filter_publish_date) &
                                                 (df_trials_candidates['Evaluation Type'] == 'pre-clinical')]
    df_trials_preclinical_count = df_trials_preclinical.groupby(['Platform']).count().reset_index()

    metric_row(
        {
            'COVID-LEAP Research Cockpit': 'Overview',
            'Total Clinical Candidates': len(df_trials_clinical),
            'Total Preclinical Candidates': len(df_trials_preclinical)
        }
    )

    # Clinical
    st.subheader('Number of Clinical Candidates by Vaccine Platform/Type')
    st.vega_lite_chart(df_trials_clinical_count, get_spec_overview_bar_chart(platform_map), use_container_width=True)

    # Preclinical
    st.subheader('Number of PreClinical Candidates by Vaccine Platform/Type')
    st.vega_lite_chart(df_trials_preclinical_count, get_spec_overview_bar_chart(platform_map), use_container_width=True)


def page_vaccine_clinical_candidates(state):
    _df = pd.DataFrame()
    default_columns = []
    subheader_text = ''

    details_view_options = [('Latest Development Stage', 'Latest Development Stage Details'),
                            ('Cumulative', 'Cumulative Clinical Trial Details'),
                            ('Candidates Only', 'Candidates Only')]

    fields_publish_date = unique_publish_date(df_trials_candidates)
    state.filter_publish_date = st.sidebar.selectbox('Select WHO Publication Date', fields_publish_date,
                                                     fields_publish_date.index(state.filter_publish_date)
                                                     if state.filter_publish_date else 0)

    state.filter_candidate_count_format = st.sidebar.selectbox('Candidate Count Format', ['Number', 'Percent'])

    state.filter_clinical_details_view = st.sidebar.selectbox('Select Details View', [o_key[0] for o_key in details_view_options])

    df_trials_clinical_by_date = df_trials_candidates[(df_trials_candidates['Publish Date'] == state.filter_publish_date)
                                                      & (df_trials_candidates['Evaluation Type'] == 'clinical')]

    metric_row(
        {
            'COVID-19 Research Cockpit': 'Clinical Candidates',
            'Total Clinical Candidates': len(df_trials_clinical_by_date)
        }
    )

    if state.filter_candidate_count_format == 'Percent':
        _df = df_trials_clinical_by_date.groupby(['Phase', 'Platform']).agg({'Id': 'count'})
        _df.reset_index(inplace=True)
        _df['Id'] = ((_df['Id'] / len(_df)) * 100).round(1)
        st.subheader('Percent of Clinical Candidates by Development Phase & Vaccine Platform/Type')
        st.vega_lite_chart(_df, get_spec_clinical_stacked_bar_h_pct(platform_map), use_container_width=True)
    else:
        st.subheader('Number of Clinical Candidates by Development Phase & Vaccine Platform/Type')
        st.vega_lite_chart(df_trials_clinical_by_date, get_spec_clinical_stacked_bar_h(platform_map), use_container_width=True)

    # Trials Details
    df_trials_clinical_stage_by_date = df_trials_clinical_stage[
        df_trials_clinical_stage['Publish Date'] == state.filter_publish_date]

    if state.filter_clinical_details_view == details_view_options[0][0]:  # Latest Dev Stage
        subheader_text = details_view_options[0][1]
        _df = df_trials_clinical_by_date.merge(df_trials_clinical_stage_by_date, left_on=['Id', 'Phase'], right_on=['Id', 'Phase'])
        _df.rename(columns={'Publish Date_x': 'Publish Date'}, inplace=True)
        _df.drop(columns=['Publisher_x', 'Publisher_y', 'Publish Date_y'], axis=1, inplace=True)
        default_columns = ['Developer', 'Phase', 'Number Doses', 'Timing Doses', 'Trial Id', 'Platform',
                           'Route of Administration', 'Type', 'Publish Date']
    elif state.filter_clinical_details_view == details_view_options[1][0]:  # Cumulative
        subheader_text = details_view_options[1][1]
        _df = df_trials_clinical_by_date.merge(df_trials_clinical_stage_by_date, left_on=['Id'], right_on=['Id'])
        _df.rename(columns={'Publish Date_x': 'Publish Date', 'Phase_y': 'Phase'}, inplace=True)
        _df.drop(columns=['Publisher_x', 'Publisher_y', 'Publish Date_y', 'Phase_x', 'Disease'], axis=1, inplace=True)
        default_columns = ['Developer', 'Phase', 'Number Doses', 'Timing Doses', 'Trial Id', 'Platform',
                           'Route of Administration', 'Type', 'Publish Date']
    elif state.filter_clinical_details_view == details_view_options[2][0]:  # Candidates
        subheader_text = details_view_options[2][1]
        _df = df_trials_clinical_by_date
        default_columns = ['Developer', 'Platform', 'Type']

    st.subheader(subheader_text + ' (' + str(len(_df)) + ')')
    cols = st.multiselect('', _df.columns.tolist(), default=default_columns)
    pd.set_option('max_colwidth', 400)

    pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
    st.dataframe(_df[cols])


def page_vaccine_clinical_candidates_to_date(state):
    df_trials_clinical = df_trials_candidates[df_trials_candidates['Evaluation Type'] == 'clinical'].groupby(
        ['Publish Date', 'Platform']).count().reset_index()

    metric_row(
        {
            'COVID-19 Research Cockpit': 'Clinical Candidates To Date',
            'WHO Publications Extracted': df_trials_clinical['Publish Date'].nunique()
        }
    )

    st.vega_lite_chart(df_trials_clinical, get_spec_clinical_multi_series_line_i(platform_map), use_container_width=True)


def page_vaccine_preclinical_candidates(state):
    fields_publish_date = unique_publish_date(df_trials_candidates)
    state.filter_publish_date = st.sidebar.selectbox('Select Publish Date', fields_publish_date,
                                                     fields_publish_date.index(state.filter_publish_date)
                                                     if state.filter_publish_date else 0)

    state.filter_candidate_count_format = st.sidebar.selectbox('Candidate Count Format', ['Number', 'Percent'])

    df_trials_preclinical_by_date = df_trials_candidates[(df_trials_candidates['Publish Date'] == state.filter_publish_date) &
                                                 (df_trials_candidates['Evaluation Type'] == 'pre-clinical')]
    df_trials_preclinical_count = df_trials_preclinical_by_date.groupby(['Platform']).count().reset_index()

    metric_row(
        {
            'COVID-19 Research Cockpit': 'Preclinical Candidates',
            'Total Preclinical Candidates': len(df_trials_preclinical_by_date)
        }
    )

    if state.filter_candidate_count_format == 'Percent':
        df_trials_preclinical_count['Id'] = ((df_trials_preclinical_count['Id'] / df_trials_preclinical_count['Id'].sum()) * 100).round(1)
        st.subheader('Percent of Preclinical Candidates by Vaccine Platform/Type')
        st.vega_lite_chart(df_trials_preclinical_count, get_spec_overview_bar_chart_pct(platform_map), use_container_width=True)
    else:
        st.subheader('Number of Preclinical Candidates by Vaccine Platform/Type')
        st.vega_lite_chart(df_trials_preclinical_count, get_spec_overview_bar_chart(platform_map), use_container_width=True)

    #st.vega_lite_chart(df_trials_preclinical_count, get_spec_overview_bar_chart(platform_map), use_container_width=True)

    st.subheader('Preclinical Trial Details')

    defaultcols = ['Developer', 'Platform', 'Type']
    cols = st.multiselect('', df_trials_preclinical_by_date.columns.tolist(), default=defaultcols)
    st.dataframe(df_trials_preclinical_by_date[cols])


def page_info(state):
    metric_row(
        {
            'COVID-LEAP Research Cockpit': 'Info',
        }
    )

    st.write('This interactive dashboard allows you to explore preclinical and clinical COVID-19 vaccines candidates \
             as tracked by the World Health Organisation.')

    st.write('Data is sourced from their landscape vaccine publication.')

    st.write('As the WHO do not maintain an archive of previous publication versions, these have been sourced from \
             web.archive.gov')

    st.write('For trials tracked by the WHO, supplementary data is also sourced from 2 additional sources:')

    st.write('* ClinicalTrials.Gov (search term COVID-19 OR covid19 OR Coronavirus for condition or disease)')
    st.write('* Aggregate Analysis of ClinicalTrials.Gov (AACT) database')


def display_state_values(state):
    pass
    # st.write("Input state:", state.input)
    # st.write("Slider state:", state.slider)
    # st.write("Radio state:", state.radio)
    # st.write("Checkbox state:", state.checkbox)
    # st.write("Selectbox state:", state.selectbox)
    # st.write("Multiselect state:", state.multiselect)
    #
    # for i in range(3):
    #     st.write(f"Value {i}:", state[f"State value {i}"])
    #
    # if st.button("Clear state"):
    #     state.clear()


if __name__ == "__main__":
    main()




