import argparse
import os
from azureml.core import Run
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine, exc
from azure.storage.blob import BlobServiceClient
import zipfile
import glob


class ETL:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.df = None
        self.articles_in = 0
        self.api = None

        self.dataset_title = 'COVID-19 Open Research Dataset Challenge (CORD-19)'
        self.dataset_ref = 'allen-institute-for-ai/CORD-19-research-challenge'
        self.dataset_ownerRef = 'allen-institute-for-ai'

        self.dataset = {'ref': None,
                        'lastUpdated': None}

        self.db_conn_path = None
        self.sqlalchemy_engine = None

        self.blob_connect_string = None

        self.psql_host = None
        self.psql_user = None
        self.psql_pwd = None
        self.dbname = 'analytics-shared-repository'
        self.get_os_environ_variables()

        self.set_db_conn_path()
        self.set_sqlalchemy_engine()

        self.get_dataset_control()

        #self.get_runtime_arguments()

        self.kaggle_authenticate()
        self.check_dataset_version()

        self.collect_metrics_pre()
        self.discovery()
        self.log_metrics_post()

    def get_os_environ_variables(self):
        print('--- Getting OS Environment Variables')
        self.psql_host = os.environ.get('AZ_RP_PSQL_HOST')
        self.psql_user = os.environ.get('AZ_RP_PSQL_USER')
        self.psql_pwd = os.environ.get('AZ_RP_PSQL_PWD')

        self.blob_connect_string = os.environ.get('AZ_RP_MLW_BLOB_CONNECT_STRING')

    def set_db_conn_path(self):
        print('--- Setting Db Connection')
        self.db_conn_path = 'postgresql://' + self.psql_user + ':' + self.psql_pwd + '@' + self.psql_host + ':5432/' + \
                            self.dbname

    def set_sqlalchemy_engine(self):
        print('--- Setting Db Engine')
        self.sqlalchemy_engine = create_engine(self.db_conn_path, connect_args={'sslmode': 'require'})

    def get_dataset_control(self):
        sql_string = 'select ref, "lastUpdated" from dataset_control where ref = ' + "'" + self.dataset_ref + "'"
        with self.sqlalchemy_engine.connect() as conn:
            rs = conn.execute(sql_string)

        for row in rs:
            try:
                self.dataset['ref'] = row[0]
                self.dataset['lastUpdated'] = row[1]
            except:
                pass

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input',
            type=str,
            help='Input extract data'
        )

        self.args = parser.parse_args()

        print('Input: {}'.format(self.args.input))

    def kaggle_authenticate(self):
        {"username": "xxx", "key": "xxx}
        self.api = KaggleApi()
        self.api.authenticate()

    def upload_file(self, client, source, dest):
        '''
        Upload a single file to a path inside the container
        '''
        print(f'Uploading {source} to {dest}')
        with open(source, 'rb') as data:
            client.upload_blob(name=dest, data=data)

    def check_dataset_version(self):
        container_name = 'datasets'
        blob_name = 'test'
        blob_service_client = BlobServiceClient.from_connection_string(self.blob_connect_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        source = 'download'

        #To delete existing blob
        #block_blob_service.delete_blob(container, blob_list[1], snapshot=None)

        datasets = self.api.dataset_list(search='allen-institute-for-ai/CORD-19-research-challenge')
        for d in datasets:
            if d.ref == self.dataset_ref:
                if d.lastUpdated > self.dataset['lastUpdated']:
                    if not os.path.isdir('download'):
                        self.api.dataset_download_files(self.dataset_ref, path='download')
                        download = glob.glob(f'download/*.*', recursive=True)
                        for idx, entry in enumerate(download):
                            with zipfile.ZipFile(entry, "r") as zip_ref:
                                zip_ref.extractall("download")
                    prefix = '' if blob_name == '' else blob_name + '/'
                    #prefix += os.path.basename(source) + '/'
                    prefix += '/'
                    for root, dirs, files in os.walk("download"):
                        for name in files:
                            if 'zip' in name:
                                continue
                            if 'cord_19_embeddings' in name:
                                continue
                            if 'Kaggle' in name:
                                continue
                            print('Uploading ', name)
                            dir_part = os.path.relpath(root, source)
                            dir_part = '' if dir_part == '.' else dir_part + '/'
                            file_path = os.path.join(root, name)
                            blob_path = prefix + dir_part + name
                            #self.upload_file(blob_client, file_path, blob_path)
                            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
                            with open(file_path, "rb") as data:
                                blob_client.upload_blob(data=data, blob_type="BlockBlob")
                print(d.creatorName)
        print(datasets)


if __name__ == "__main__":
    print('--- Kaggle CORD-19')
    etl = ETL()
    print('--- Kaggle CORD-19 Completed')
