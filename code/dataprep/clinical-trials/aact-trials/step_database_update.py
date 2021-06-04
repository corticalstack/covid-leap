import argparse
import os
import re
import pandas as pd
import numpy as np
from azureml.core import Run
from sqlalchemy import create_engine


class DatabaseUpdate:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.get_runtime_arguments()

        self.path = None

        self.studies_in = 0
        self.studies_out = 0

        self.design_outcomes_in = 0
        self.design_outcomes_out = 0

        self.links_in = 0
        self.links_out = 0

        self.calculated_values_in = 0
        self.calculated_values_out = 0

        self.countries_in = 0
        self.countries_out = 0

        self.sponsors_in = 0
        self.sponsors_out = 0

        self.design_groups_in = 0
        self.design_groups_out = 0

        self.interventions_in = 0
        self.interventions_out = 0

        self.eligibilities_in = 0
        self.eligibilities_out = 0

        self.brief_summaries_in = 0
        self.brief_summaries_out = 0

        self.custom_calculated_values_out = 0

        self.df_ref = pd.DataFrame()
        self.df_ref_id = pd.DataFrame()

        self.db_conn_path = None
        self.sqlalchemy_engine = None

        self.psql_host = None
        self.psql_user = None
        self.psql_pwd = None
        self.dbname = 'analytics-shared-repository'
        self.get_os_environ_variables()

        self.set_db_conn_path()
        self.set_sqlalchemy_engine()

        self.update_db_trials()

        self.log_metrics_post()

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input',
            type=str,
            help='Path to input data'
        )
        parser.add_argument(
            '--extract_terms',
            type=str,
            help='ClinicalTrialsGov extract terms'
        )

        self.args = parser.parse_args()

        print('Input: {}'.format(self.args.input))
        print('ClinicalTrialsGov Extract Terms: {}'.format(self.args.extract_terms))

    def get_os_environ_variables(self):
        print('--- Getting OS Environment Variables')
        self.psql_host = os.environ.get('AZ_RP_PSQL_HOST')
        self.psql_user = os.environ.get('AZ_RP_PSQL_USER')
        self.psql_pwd = os.environ.get('AZ_RP_PSQL_PWD')

    def set_db_conn_path(self):
        print('--- Setting Db Connection')
        self.db_conn_path = 'postgresql://' + self.psql_user + ':' + self.psql_pwd + '@' + self.psql_host + ':5432/' + \
                            self.dbname

    def set_sqlalchemy_engine(self):
        print('--- Setting Db Engine')
        self.sqlalchemy_engine = create_engine(self.db_conn_path, connect_args={'sslmode': 'require'})

    def studies(self, path, table_name):
        print('--- Processing Studies')

        self.df_ref = pd.read_csv(path, sep='|', dtype={
            'nct_id': str,
            'nlm_download_date_description': str,
            'study_first_submitted_date': str,
            'results_first_submitted_date': str,
            'disposition_first_submitted_date': str,
            'last_update_submitted_date': str,
            'study_first_submitted_qc_date': str,
            'study_first_posted_date': str,
            'study_first_posted_date_type': str,
            'results_first_submitted_qc_date': str,
            'results_first_posted_date': str,
            'results_first_posted_date_type': str,
            'disposition_first_submitted_qc_date': str,
            'disposition_first_posted_date': str,
            'disposition_first_posted_date_type': str,
            'last_update_submitted_qc_date': str,
            'last_update_posted_date': str,
            'last_update_posted_date_type': str,
            'start_month_year': str,
            'start_date_type': str,
            'start_date': str,
            'verification_month_year': str,
            'verification_date': str,
            'completion_month_year': str,
            'completion_date_type': str,
            'completion_date': str,
            'primary_completion_month_year': str,
            'primary_completion_date_type': str,
            'primary_completion_date': str,
            'target_duration': str,
            'study_type': str,
            'acronym': str,
            'baseline_population': str,
            'brief_title': str,
            'official_title': str,
            'overall_status': str,
            'last_known_status': str,
            'phase': str,
            'enrollment': str,
            'enrollment_type': str,
            'source': str,
            'limitations_and_caveats': str,
            'number_of_arms': str,
            'number_of_groups': str,
            'why_stopped': str,
            'has_expanded_access': str,
            'expanded_access_type_individual': str,
            'expanded_access_type_intermediate': str,
            'expanded_access_type_treatment': str,
            'has_dmc': str,
            'is_fda_regulated_drug': str,
            'is_fda_regulated_device': str,
            'is_unapproved_device': str,
            'is_ppsd': str,
            'is_us_export': str,
            'biospec_retention': str,
            'biospec_description': str,
            'ipd_time_frame': str,
            'ipd_access_criteria': str,
            'ipd_url': str,
            'plan_to_share_ipd': str,
            'plan_to_share_ipd_description': str,
            'created_at': str,
            'updated_at': str
        })

        self.studies_in += self.df_ref.shape[0]

        self.df_ref = self.df_ref.replace(np.nan, '')

        # Transform
        self.df_ref['study_first_submitted_date'] =  pd.to_datetime(self.df_ref['study_first_submitted_date'], errors='ignore').dt.date
        self.df_ref['results_first_submitted_date'] = pd.to_datetime(self.df_ref['results_first_submitted_date'], errors='ignore').dt.date
        self.df_ref['disposition_first_submitted_date'] = pd.to_datetime(self.df_ref['disposition_first_submitted_date'], errors='ignore').dt.date
        self.df_ref['last_update_submitted_date'] = pd.to_datetime(self.df_ref['last_update_submitted_date'], errors='ignore').dt.date
        self.df_ref['study_first_submitted_qc_date'] = pd.to_datetime(self.df_ref['study_first_submitted_qc_date'], errors='ignore').dt.date
        self.df_ref['study_first_posted_date'] = pd.to_datetime(self.df_ref['study_first_posted_date'], errors='ignore').dt.date
        self.df_ref['results_first_submitted_qc_date'] = pd.to_datetime(self.df_ref['results_first_submitted_qc_date'], errors='ignore').dt.date
        self.df_ref['results_first_posted_date'] = pd.to_datetime(self.df_ref['results_first_posted_date'], errors='ignore').dt.date
        self.df_ref['disposition_first_submitted_qc_date'] = pd.to_datetime(self.df_ref['disposition_first_submitted_qc_date'], errors='ignore').dt.date
        self.df_ref['disposition_first_posted_date'] = pd.to_datetime(self.df_ref['disposition_first_posted_date'], errors='ignore').dt.date
        self.df_ref['last_update_submitted_qc_date'] = pd.to_datetime(self.df_ref['last_update_submitted_qc_date'], errors='ignore').dt.date
        self.df_ref['last_update_posted_date'] = pd.to_datetime(self.df_ref['last_update_posted_date'], errors='ignore').dt.date
        self.df_ref['start_date'] = pd.to_datetime(self.df_ref['start_date'], errors='ignore').dt.date
        self.df_ref['verification_date'] = pd.to_datetime(self.df_ref['verification_date'], errors='ignore').dt.date
        self.df_ref['completion_date'] = pd.to_datetime(self.df_ref['completion_date'], errors='ignore').dt.date
        self.df_ref['primary_completion_date'] = pd.to_datetime(self.df_ref['primary_completion_date'], errors='ignore').dt.date
        self.df_ref['enrollment'] = pd.to_numeric(self.df_ref['enrollment'])
        self.df_ref['number_of_arms'] = pd.to_numeric(self.df_ref['number_of_arms'])
        self.df_ref['number_of_groups'] = pd.to_numeric(self.df_ref['number_of_groups'])
        self.df_ref['created_at'] = pd.to_datetime(self.df_ref['created_at'], errors='ignore')
        self.df_ref['created_at'] = pd.to_datetime(self.df_ref['created_at'], errors='ignore')
        self.df_ref['updated_at'] = pd.to_datetime(self.df_ref['updated_at'])

        search_text = 'COVID-19|covid|sar cov 2|SARS-CoV-2|2019-nCov|2019 ncov|SARS Coronavirus 2|2019 Novel Coronavirus' \
                      '|coronavirus 2019| Wuhan coronavirus|wuhan pneumonia|wuhan virus|China virus'
        print('NOTE! - AACT trials dataset filtered by {}'.format(search_text))

        self.df_ref = self.df_ref[self.df_ref.official_title.str.contains(search_text, case=False) |
                                  self.df_ref.brief_title.str.contains(search_text, case=False)]

        self.df_ref_id = self.df_ref[['nct_id']].copy()

        self.update_db(self.df_ref, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'nct_id')
            conn.execute(sql_string)

        self.studies_out += self.df_ref.shape[0]

    def design_outcomes(self, path, table_name):
        print('--- Processing Design Outcomes')

        df = pd.read_csv(path, sep='|', dtype={
            'id': int,
            'nct_id': str,
            'outcome_type': str,
            'measure': str,
            'time_frame': str,
            'population': str,
            'description': str
        })

        self.design_outcomes_in += df.shape[0]

        df = df.replace(np.nan, '')

        df_temp = self.df_ref_id.merge(df, left_on='nct_id', right_on='nct_id')

        self.update_db(df_temp, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'id')
            conn.execute(sql_string)

        self.design_outcomes_out += df_temp.shape[0]

    def links(self, path, table_name):
        print('--- Processing Links')

        df = pd.read_csv(path, sep='|', dtype={
            'id': int,
            'nct_id': str,
            'url': str,
            'description': str
        })

        self.links_in = df.shape[0]

        df = df.replace(np.nan, '')

        df_temp = self.df_ref_id.merge(df, left_on='nct_id', right_on='nct_id')
        self.update_db(df_temp, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'id')
            conn.execute(sql_string)

        self.links_out += df_temp.shape[0]

    def countries(self, path, table_name):
        print('--- Processing Countries')

        df = pd.read_csv(path, sep='|', dtype={
            'id': int,
            'nct_id': str,
            'name': str,
            'removed': str
        })

        self.countries_in += df.shape[0]

        df = df.replace(np.nan, '')

        df_temp = self.df_ref_id.merge(df, left_on='nct_id', right_on='nct_id')
        self.update_db(df_temp, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'id')
            conn.execute(sql_string)

        self.countries_out += df_temp.shape[0]

    def calculated_values(self, path, table_name):
        print('--- Processing Calculated Values')

        df = pd.read_csv(path, sep='|', dtype={
            'id': int,
            'nct_id': str,
            'number_of_facilities': str,
            'number_of_nsae_subjects': str,
            'number_of_sae_subjects': str,
            'registered_in_calendar_year': str,
            'nlm_download_date': str,
            'actual_duration': str,
            'were_results_reported': str,
            'months_to_report_results': str,
            'has_us_facility': str,
            'has_single_facility': str,
            'minimum_age_num': str,
            'maximum_age_num': str,
            'minimum_age_unit': str,
            'maximum_age_unit': str,
            'number_of_primary_outcomes_to_measure': str,
            'number_of_secondary_outcomes_to_measure': str,
            'number_of_other_outcomes_to_measure': str
        })

        self.calculated_values_in += df.shape[0]

        df = df.replace(np.nan, '')

        df['nlm_download_date'] = pd.to_datetime(df['nlm_download_date'], errors='ignore').dt.date
        df['number_of_facilities'] = pd.to_numeric(df['number_of_facilities'])
        df['number_of_nsae_subjects'] = pd.to_numeric(df['number_of_nsae_subjects'])
        df['number_of_sae_subjects'] = pd.to_numeric(df['number_of_sae_subjects'])
        df['number_of_facilities'] = pd.to_numeric(df['number_of_facilities'])
        df['actual_duration'] = pd.to_numeric(df['actual_duration'])
        df['months_to_report_results'] = pd.to_numeric(df['months_to_report_results'])
        df['minimum_age_num'] = pd.to_numeric(df['minimum_age_num'])
        df['maximum_age_num'] = pd.to_numeric(df['maximum_age_num'])
        df['number_of_primary_outcomes_to_measure'] = pd.to_numeric(df['number_of_primary_outcomes_to_measure'])
        df['number_of_secondary_outcomes_to_measure'] = pd.to_numeric(df['number_of_secondary_outcomes_to_measure'])
        df['number_of_other_outcomes_to_measure'] = pd.to_numeric(df['number_of_other_outcomes_to_measure'])

        df_temp = self.df_ref_id.merge(df, left_on='nct_id', right_on='nct_id')
        self.update_db(df_temp, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'id')
            conn.execute(sql_string)

        self.calculated_values_out += df_temp.shape[0]

    def sponsors(self, path, table_name):
        print('--- Processing Sponsors')

        df = pd.read_csv(path, sep='|', dtype={
            'id': int,
            'nct_id': str,
            'agency_class': str,
            'lead_or_collaborator': str,
            'name': str
        })

        self.sponsors_in += df.shape[0]

        df = df.replace(np.nan, '')

        df_temp = self.df_ref_id.merge(df, left_on='nct_id', right_on='nct_id')
        self.update_db(df_temp, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'id')
            conn.execute(sql_string)

        self.sponsors_out += df_temp.shape[0]

    def interventions(self, path, table_name):
        print('--- Processing Interventions')

        df = pd.read_csv(path, sep='|', dtype={
            'id': int,
            'nct_id': str,
            'intervention_type': str,
            'name': str,
            'description': str
        })

        self.interventions_in += df.shape[0]

        df = df.replace(np.nan, '')

        df_temp = self.df_ref_id.merge(df, left_on='nct_id', right_on='nct_id')
        self.update_db(df_temp, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'id')
            conn.execute(sql_string)

        self.interventions_out += df_temp.shape[0]

    def eligibilities(self, path, table_name):
        print('--- Processing Eligibilities')

        df = pd.read_csv(path, sep='|', dtype={
            'id': int,
            'nct_id': str,
            'sampling_method': str,
            'gender': str,
            'minimum_age': str,
            'maximum_age': str,
            'healthy_volunteers': str,
            'population': str,
            'criteria': str,
            'gender_description': str,
            'gender_based': str
        })

        self.eligibilities_in += df.shape[0]

        df = df.replace(np.nan, '')

        df_temp = self.df_ref_id.merge(df, left_on='nct_id', right_on='nct_id')
        self.update_db(df_temp, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'id')
            conn.execute(sql_string)

        self.eligibilities_out += df_temp.shape[0]

    def design_groups(self, path, table_name):
        print('--- Processing Design Groups')

        df = pd.read_csv(path, sep='|', dtype={
            'id': int,
            'nct_id': str,
            'group_type': str,
            'title': str,
            'description': str
        })

        self.design_groups_in += df.shape[0]

        df = df.replace(np.nan, '')

        df_temp = self.df_ref_id.merge(df, left_on='nct_id', right_on='nct_id')
        self.update_db(df_temp, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'id')
            conn.execute(sql_string)

        self.design_groups_out += df_temp.shape[0]

    def brief_summaries(self, path, table_name):
        print('--- Processing Brief Summaries')

        df = pd.read_csv(path, sep='|', dtype={
            'id': int,
            'nct_id': str,
            'description': str
        })

        self.brief_summaries_in += df.shape[0]

        df = df.replace(np.nan, '')

        df_temp = self.df_ref_id.merge(df, left_on='nct_id', right_on='nct_id')
        self.update_db(df_temp, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'id')
            conn.execute(sql_string)

        self.brief_summaries_out += df_temp.shape[0]

    def custom_calculated_values(self):
        print('--- Processing Custom Calculated Values')
        subjects = self.custom_subjects()
        primary_completion_date = self.custom_primary_completion_date()
        vaccine_name = self.custom_vaccine_name()

        df_subjects = pd.DataFrame.from_dict(subjects, orient='index')
        df_subjects.reset_index(inplace=True)
        df_subjects.columns = ['nct_id', 'subjects']

        df_primary_completion_date = pd.DataFrame.from_dict(primary_completion_date, orient='index')
        df_primary_completion_date.reset_index(inplace=True)
        df_primary_completion_date.columns = ['nct_id', 'primary_completion_date']

        df_vaccine_name = pd.DataFrame.from_dict(vaccine_name, orient='index')
        df_vaccine_name.reset_index(inplace=True)
        df_vaccine_name.columns = ['nct_id', 'vaccine_name']

        df_custom = pd.merge(left=df_subjects, right=df_primary_completion_date, how='outer')
        df_custom = pd.merge(left=df_custom, right=df_vaccine_name, how='outer')

        df_custom.fillna('', inplace=True)
        self.update_db(df_custom, 'aact_custom_values')

        self.custom_calculated_values_out += df_custom.shape[0]

    def custom_subjects(self):
        re_subjects = ['\d\d-\d\d\syears', '\d\d to \d\d\syears', 'adults(.*?)up', 'between(.*?)old', 'aged(.*?)old',
                   'aged(.*?)years',
                   '\d\d years(.*?)older', '\((.*?)older\)', '<(.*?)old', 'Moderate(.*?)years', '\S\S+ years',
                   'subjects (.*?)years(.*?)older', 'participants\(\d\d-\d\d', 'participants\(\d-\d\d',
                   'participants\(\d\d years of age and above']
        combined_re_subjects = "(" + ")|(".join(re_subjects) + ")"

        # Derive subjects
        subjects = {}

        sql_string = "select nct_id, title, description from aact_design_groups where title like lower('%%years%%') or title like " \
                     "lower('%%age%%') or title like lower('%%aged%%') or title like lower('%%participants%%')" \
                     "or description like lower('%%years%%') or description like lower('%%age%%') or description like " \
                     "lower('%%aged%%') or description like lower('%%participants%%')"
        with self.sqlalchemy_engine.connect() as conn:
            rs = conn.execute(sql_string)

        for row in rs:
            try:
                # Design Groups Title
                found = re.search(combined_re_subjects, row[1], re.IGNORECASE).group()

                if row[0] in subjects:
                    subjects[row[0]].append(found)
                else:
                    subjects[row[0]] = [found]
            except AttributeError:
                pass
            try:
                # Design Groups Description
                found = re.search(combined_re_subjects, row[2], re.IGNORECASE).group()
                if row[0] in subjects:
                    subjects[row[0]].append(found)
                else:
                    subjects[row[0]] = [found]
            except AttributeError:
                pass

        sql_string = 'select "NCT Number", "Age" from "clinicaltrialsgov" where "extract_terms" = ' \
                     "'" + self.args.extract_terms + "'"
        with self.sqlalchemy_engine.connect() as conn:
            rs = conn.execute(sql_string)

        for row in rs:
            try:
                found = re.search(combined_re_subjects, row[1], re.IGNORECASE).group()
            except AttributeError:
                pass
            if not row[0] in subjects:
                subjects[row[0]] = [row[1]]

        sql_string = "select nct_id, minimum_age, maximum_age from aact_eligibilities"
        with self.sqlalchemy_engine.connect() as conn:
            rs = conn.execute(sql_string)

        for row in rs:
            # Subjects already derived or no age information
            if row[0] in subjects or (not row[1] and not row[2]):
                continue

            min_age = row[1].replace(' Years', '')
            max_age = row[2].replace(' Years', '')
            age_range = min_age + '-' + max_age
            subjects[row[0]] = [age_range]

        for key, values in subjects.items():
            for idx, v in enumerate(values):
                subjects[key][idx] = subjects[key][idx].replace('participants', '')
                subjects[key][idx] = subjects[key][idx].replace('(Adult)', '')
                subjects[key][idx] = subjects[key][idx].replace('(Child, Adult, Older Adult)', '')
                subjects[key][idx] = subjects[key][idx].replace('(Adult, Older Adult)', '')
                subjects[key][idx] = subjects[key][idx].replace('(', '')
                subjects[key][idx] = subjects[key][idx].replace(')', '')
                subjects[key][idx] = subjects[key][idx].replace(' Years to ', '-')
                subjects[key][idx] = subjects[key][idx].replace(' to ', '-')
                subjects[key][idx] = subjects[key][idx].replace('aged', '')
                subjects[key][idx] = subjects[key][idx].replace('Aged', '')
                subjects[key][idx] = subjects[key][idx].replace('years', '')
                subjects[key][idx] = subjects[key][idx].replace('of age', '')
                subjects[key][idx] = subjects[key][idx].replace('subjects', '')
                subjects[key][idx] = subjects[key][idx].strip()
                subjects[key][idx] = re.sub(' +', ' ', subjects[key][idx])

            subjects[key] = list(set(values))
            subjects[key].sort()
            subjects[key] = ', '.join(subjects[key])
        return subjects

    def custom_primary_completion_date(self):
        primary_completion_date = {}

        sql_string = 'select "nct_id", "primary_completion_date" from "aact_studies" where "primary_completion_date" ' \
                     'is not null'
        with self.sqlalchemy_engine.connect() as conn:
            rs = conn.execute(sql_string)

        for row in rs:
            if row[1] is None:
                continue
            primary_completion_date[row[0]] = row[1]

        sql_string = 'select "NCT Number", "Primary Completion Date"::TIMESTAMP::DATE from clinicaltrialsgov'

        with self.sqlalchemy_engine.connect() as conn:
            rs = conn.execute(sql_string)

        for row in rs:
            if row[1] is None:
                continue
            if not row[0] in primary_completion_date:
                primary_completion_date[row[0]] = row[1]

        return primary_completion_date

    def custom_vaccine_name(self):
        re_omit = ['(.*?)placebo(.*?)', '(.*?)standard of care(.*?)', '(.*?)control(.*?)', '(.*?)blood(.*?)',
                   '(.*?)natural(.*?)', '(.*?)standard(.*?)', '(.*?)supportive(.*?)', '(.*?)control(.*?)',
                   '(.*?)stools(.*?)', '(.*?)sample(.*?)', '(.*?)patch(.*?)', '(.*?)serology(.*?)']
        combined_re_omit = "(" + ")|(".join(re_omit) + ")"

        vaccine_name = {}

        sql_string = 'select "nct_id", "intervention_type", "name" from "aact_interventions" ' \
        "where intervention_type in ('Biological', 'Drug')"
        with self.sqlalchemy_engine.connect() as conn:
            rs = conn.execute(sql_string)

        for row in rs:
            try:
                re.search(combined_re_omit, row[2], re.IGNORECASE).group()
            except AttributeError:
                if row[0] in vaccine_name:
                    vaccine_name[row[0]].append(row[2])
                else:
                    vaccine_name[row[0]] = [row[2]]

        for key, values in vaccine_name.items():
            for idx, v in enumerate(values):
                vaccine_name[key][idx] = re.sub(' +', ' ', vaccine_name[key][idx])

            vaccine_name[key] = list(set(values))
            vaccine_name[key].sort()
            vaccine_name[key] = ', '.join(vaccine_name[key])

        return vaccine_name

    def update_db(self, df, table_name):
        df.to_sql(table_name, self.sqlalchemy_engine, index=False, method='multi', if_exists='replace')

    def update_db_trials(self):
        print('--- Update Db Trials')

        files = ['studies.txt', 'design_outcomes.txt', 'links.txt', 'countries.txt', 'calculated_values.txt',
                 'sponsors.txt', 'interventions.txt', 'eligibilities.txt', 'design_groups.txt', 'brief_summaries.txt']

        self.custom_calculated_values()

        for file_name in files:
            path = f'{self.args.input}/{file_name}'
            file = file_name.split('.')
            table_name = 'aact_' + file[0]
            try:
                method = getattr(self, file[0])
            except AttributeError:
                print('Database injection of {} not currently supported'.format(file_name))
            method(path, table_name)

    def log_metrics_post(self):
        self.run.log('# Studies in', self.studies_in)
        self.run.log('# Studies out', self.studies_out)

        self.run.log('# Design outcomes in', self.design_outcomes_in)
        self.run.log('# Design outcomes out', self.design_outcomes_out)

        self.run.log('# Links in', self.links_in)
        self.run.log('# Links out', self.links_out)

        self.run.log('# Calculated values in', self.calculated_values_in)
        self.run.log('# Calculated values out', self.calculated_values_out)

        self.run.log('# Countries in', self.countries_in)
        self.run.log('# Countries out', self.countries_out)

        self.run.log('# Sponsors in', self.sponsors_in)
        self.run.log('# Sponsors out', self.sponsors_out)

        self.run.log('# Design groups in', self.design_groups_in)
        self.run.log('# Design groups out', self.design_groups_out)

        self.run.log('# Interventions in', self.interventions_in)
        self.run.log('# Interventions out', self.interventions_out)

        self.run.log('# Eligibilities in', self.eligibilities_in)
        self.run.log('# Eligibilities out', self.eligibilities_out)

        self.run.log('# Brief summaries in', self.brief_summaries_in)
        self.run.log('# Brief summaries out', self.brief_summaries_out)

        self.run.log('# Custom calculated values out', self.custom_calculated_values_out)


if __name__ == "__main__":
    print('--- Database Update')
    database_update = DatabaseUpdate()
    print('--- Database Update Completed')
