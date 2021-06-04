import argparse
import os
import pandas as pd
import numpy as np
from azureml.core import Run
from sqlalchemy import create_engine
import glob


class DatabaseUpdate:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.get_runtime_arguments()

        self.all_extracts = None
        self.articles_in = 0
        self.articles_out = 0
        
        self.path = None

        self.df = pd.DataFrame()
        self.df_id = pd.DataFrame()

        self.db_conn_path = None
        self.sqlalchemy_engine = None

        self.psql_host = None
        self.psql_user = None
        self.psql_pwd = None
        self.dbname = 'analytics-shared-repository'
        self.get_os_environ_variables()

        self.set_db_conn_path()
        self.set_sqlalchemy_engine()

        self.ctg_extracts()

        self.log_metrics_post()

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
        self.db_conn_path = 'postgresql://' + self.psql_user + ':' + self.psql_pwd + '@' + self.psql_host + ':5432/' + \
                            self.dbname

    def set_sqlalchemy_engine(self):
        print('--- Setting Db Engine')
        self.sqlalchemy_engine = create_engine(self.db_conn_path, connect_args={'sslmode': 'require'})

    def get_extract_list(self):
        print('--- Get Extract List')
        self.all_extracts = glob.glob(f'{self.args.input}/*.csv', recursive=True)
        print('Number CTG Extract {}'.format(str(len(self.all_extracts))))

    def ctg_extracts(self):
        print('--- Processing CTG Extracts')
        
        self.get_extract_list()

        for idx, entry in enumerate(self.all_extracts):
            print('\nProcessing CTG extract {}'.format(entry))
            
            self.df = pd.read_csv(entry, sep=',', dtype={
                'Rank': str,
                'NCT Number': str,
                'Title': str,
                'Acronym': str,
                'Status': str,
                'Study Results': str,
                'Conditions': str,
                'Interventions': str,
                'Outcome Measures': str,
                'Sponsor/Collaborators': str,
                'Gender': str,
                'Age': str,
                'Phases': str,
                'Enrollment': str,
                'Funded Bys': str,
                'Study Type': str,
                'Study Designs': str,
                'Other IDs': str,
                'Start Date': str,
                'Primary Completion Date': str,
                'Completion Date': str,
                'First Posted': str,
                'Results First Posted': str,
                'Last Update Posted': str,
                'Locations': str,
                'Study Documents': str,
                'URL': str
            })

            self.articles_in += self.df.shape[0]

            self.df.drop(['Rank'], axis=1, inplace=True)
    
            self.df = self.df.replace(np.nan, '')

            self.df['extract_terms'] = entry.split('/')[-1].split('.')[0]

            self.df = self.df[self.df['NCT Number'].str.contains("NCT")]
    
            # Transform
            self.df['Start Date'] = pd.to_datetime(self.df['Start Date'], errors='ignore')
            self.df['Primary Completion Date'] = pd.to_datetime(self.df['Primary Completion Date'], errors='ignore')
            self.df['Completion Date'] = pd.to_datetime(self.df['Completion Date'], errors='ignore')
            self.df['First Posted'] = pd.to_datetime(self.df['First Posted'], errors='ignore')
            self.df['Results First Posted'] = pd.to_datetime(self.df['Results First Posted'], errors='ignore')
            self.df['Last Update Posted'] = pd.to_datetime(self.df['Last Update Posted'], errors='ignore')
    
            table_name = 'clinicaltrialsgov'
            self.update_db(self.df, table_name)
            self.articles_out += self.df.shape[0]
    
            with self.sqlalchemy_engine.connect() as conn:
                sql_string = 'ALTER TABLE "{}" ADD PRIMARY KEY ({})'.format(table_name, '"NCT Number"')
                conn.execute(sql_string)

    def update_db(self, df, table_name):
        df.to_sql(table_name, self.sqlalchemy_engine, index=False, method='multi', if_exists='replace')

    def log_metrics_post(self):
        self.run.log('# ClinicalTrails.Gov extracts', self.articles_in)
        self.run.log('# Trails to database', self.articles_out)


if __name__ == "__main__":
    print('--- Database Update')
    database_update = DatabaseUpdate()
    print('--- Database Update Completed')
