import argparse
import os
import pandas as pd
import numpy as np
import glob
import json
import re
from azureml.core import Run
from sqlalchemy import create_engine


class DatabaseUpdate:
    def __init__(self):
        self.run = Run.get_context()

        self.path = None
        self.all_trials_json = None

        self.articles_in = 0

        self.df_candidate = pd.DataFrame(columns=['id', 'publisher', 'publish_date', 'evaluation_type', 'disease',
                                                  'manufacturer', 'platform', 'type', 'number_doses', 'timing_doses',
                                                  'route_administration', 'shared_platform', 'phase'])

        self.df_clinical_stage = pd.DataFrame(columns=['id', 'publisher', 'publish_date', 'phase', 'trial_id'])

        self.args = None
        self.get_runtime_arguments()

        self.db_conn_path = None
        self.sqlalchemy_engine = None

        self.psql_host = None
        self.psql_user = None
        self.psql_pwd = None
        self.dbname = 'analytics-shared-repository'
        self.get_os_environ_variables()

        self.set_db_conn_path()
        self.set_sqlalchemy_engine()

        self.get_trials_landscape_article_list()

        self.collect_metrics_pre()

        self.process_trials_landscape_article_list()
        self.update_db_candidates('trials_candidates')
        self.update_db_clinical_stage('trials_clinical_stage')

        self.log_metrics_post()

    @staticmethod
    def file_reader(file_path):
        with open(file_path) as file:
            return json.load(file)

    @staticmethod
    def map_platform(platform):
        p = platform.lstrip().rstrip()

        regex = '(.*?)non-replicating viral(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'Viral Vector: Non-Replicating'

        regex = '(.*?)vector .non-replicating(.*?)APC'
        if re.search(regex, p, re.IGNORECASE):
            return 'Viral Vector: Non-Replicating + APC'

        regex = '(.*?)vector .non-replicating(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'Viral Vector: Non-Replicating'

        regex = '(.*?)replicating viral(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'Viral Vector: Replicating'

        regex = '(.*?)vector .replicating(.*?)APC'
        if re.search(regex, p, re.IGNORECASE):
            return 'Viral Vector: Replicating + APC'

        regex = '(.*?)vector .replicating(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'Viral Vector: Replicating'

        regex = '(.*?)replicating bacteria(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'Vector: Bacterial, Replicating'

        regex = '(.*?)dna(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'Nucleic Acid: DNA'

        regex = '(.*?)rna(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'Nucleic Acid: RNA'

        regex = ['(.*?)live attenuated virus(.*?)', '(.*?)live attenuated bacterial(.*?)']
        regex = "(" + ")|(".join(regex) + ")"
        if re.search(regex, p, re.IGNORECASE):
            return 'SARS-CoV-2 Virus: Live Attenuated'

        regex = '(.*?)inactivated(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'SARS-CoV-2 Virus: Inactivated'

        regex = '(.*?)t-cell(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'T-cell Based'

        regex = '(.*?)virus like particle(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'Virus Like Particle'

        regex = '(.*?)vlp(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'Virus Like Particle'

        regex = '(.*?)protein subunit(.*?)'
        if re.search(regex, p, re.IGNORECASE):
            return 'Protein Subunit'

        return p

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input',
            type=str,
            help='Path to input data'
        )

        self.args = parser.parse_args()

        print('Input: {}'.format(self.args.input))

    def get_os_environ_variables(self):
        print('--- Getting OS Environment Variables')
        self.psql_host = os.environ.get('AZ_RP_PSQL_HOST')
        self.psql_user = os.environ.get('AZ_RP_PSQL_USER')
        self.psql_pwd = os.environ.get('AZ_RP_PSQL_PWD')

    def set_db_conn_path(self):
        print('--- Setting Db Connection')
        self.db_conn_path = 'postgresql://' + self.psql_user + ':' + self.psql_pwd + '@' + self.psql_host + ':5432/' \
                            + self.dbname

    def set_sqlalchemy_engine(self):
        print('--- Setting Db Engine')
        self.sqlalchemy_engine = create_engine(self.db_conn_path, connect_args={'sslmode': 'require'})

    def get_trials_landscape_article_list(self):
        print('--- Get Trials Landscape Article List')
        self.all_trials_json = glob.glob(f'{self.args.input}/*.json', recursive=True)
        print('Number trials landscape articles {}'.format(str(len(self.all_trials_json))))

    def process_trials_landscape_article_list(self):
        print('--- Process Trials Landscape Article List')
        total_cand = 0
        total_trial = 0
        for idx_f, entry in enumerate(self.all_trials_json):
            print('Processing trials article {}'.format(entry))
            content = self.file_reader(entry)
            file_name_parts = entry.split('.json')
            publish_date = file_name_parts[0][-8:]
            for idx_c, candidate in enumerate(content):
                total_cand += 1
                platform = self.map_platform(candidate['metadata']['platform'])
                self.df_candidate = self.df_candidate.append({'id': candidate['id'],
                                                              'publisher': 'WHO',
                                                              'publish_date': publish_date,
                                                              'evaluation_type': candidate['evaluation_type'],
                                                              'disease': candidate['metadata']['disease'],
                                                              'manufacturer': candidate['metadata']['manufacturer'],
                                                              'platform': platform,
                                                              'type': candidate['metadata']['type'],
                                                              'number_doses': candidate['metadata']['number_doses'],
                                                              'timing_doses': candidate['metadata']['timing_doses'],
                                                              'route_administration': candidate['metadata']['route_administration'],
                                                              'shared_platform': candidate['metadata']['shared_platform'],
                                                              'phase': candidate['metadata']['phase']}, ignore_index=True)

                if candidate['evaluation_type'] == 'clinical':
                    for trial in candidate['metadata']['clinical_stage']:
                        total_trial += 1
                        self.df_clinical_stage = self.df_clinical_stage.append({'id': candidate['id'],
                                                                                'publisher': 'WHO',
                                                                                'publish_date': publish_date,
                                                                                'phase': trial['phase'],
                                                                                'trial_id': trial['id']},
                                                                               ignore_index=True)
        print('Number candidates ', total_cand)
        print('Number stages ', total_trial)
        self.df_candidate['publish_date'] = pd.to_datetime(self.df_candidate['publish_date'], errors='ignore').dt.date
        self.df_clinical_stage['publish_date'] = pd.to_datetime(self.df_clinical_stage['publish_date'],
                                                                errors='ignore').dt.date

    def update_db_candidates(self, table_name):
        print('--- Update dB Candidates')

        self.update_db(self.df_candidate, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(table_name, 'id, publisher, publish_date')
            conn.execute(sql_string)

    def update_db_clinical_stage(self, table_name):
        print('--- Update dB Clinical Stage')

        self.update_db(self.df_clinical_stage, table_name)

        with self.sqlalchemy_engine.connect() as conn:
            sql_string = 'ALTER TABLE {} ADD PRIMARY KEY ({})'.format(
                table_name, 'id, publisher, publish_date, phase, trial_id')
            conn.execute(sql_string)

    def update_db(self, df, table_name):
        # Prevent unique index violation, as duplicate entries have been observed in original source
        # eg WHO landscape 20210302 EUCTR2021-000412-28-BE
        df.drop_duplicates(inplace=True)
        df.to_sql(table_name, self.sqlalchemy_engine, index=False, method='multi', if_exists='replace')

    def collect_metrics_pre(self):
        self.articles_in = len(self.all_trials_json)

    def log_metrics_post(self):
        self.run.log('# Trials articles evaluated', self.articles_in)
        self.run.log('# Clinical evaluations',
                     self.df_candidate[self.df_candidate.evaluation_type == 'clinical'].shape[0])
        self.run.log('# Pre-clinical evaluations',
                     self.df_candidate[self.df_candidate.evaluation_type == 'pre-clinical'].shape[0])


if __name__ == "__main__":
    print('--- Database Update')
    database_update = DatabaseUpdate()
    print('--- Database Update Completed')
