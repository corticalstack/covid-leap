import time
from pprint import pprint
import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.email_operator import EmailOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
import json
from airflow.utils.dates import days_ago
from datetime import timedelta
from azure.storage.blob import BlobServiceClient, ContainerClient


#import sys
#sys.path.insert(0, '../..')

from utilities.helper_os_environ import HelperOsEnviron
from utilities.helper_kaggle import HelperKaggle
from utilities.helper_sqlalchemy import HelperSqlalchemy

import os
import shutil



default_args = {
    'owner': 'xxx',
    'depends_on_past': False,
    'email': 'xxx',
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 0
}

dag = DAG(
    dag_id='dataset_etl_kaggle_cord19',
    default_args=default_args,
    description='Extract and load Kaggle CORD19 dataset',
    start_date=datetime.datetime(2021, 3, 31, 8),
    schedule_interval='38 * * * *',
    tags=['cord19'],
    catchup=False
)

dataset_ref = 'allen-institute-for-ai/CORD-19-research-challenge'

kaggle_dataset_info = []


def fetch_azure_cord19_date(ti):
    print('--- Fetch Azure Cord19 last updated')
    h_os_environ = HelperOsEnviron()
    h_sqlalchemy = HelperSqlalchemy(h_os_environ, 'analytics-shared-repository')
    print('Host', h_os_environ.az_rp_psql_host)
    print('User', h_os_environ.az_rp_psql_user)
    print('Pwd', h_os_environ.az_rp_psql_pwd)
    print('Kaggle username', h_os_environ.kaggle_username)
    print('Kaggle key', h_os_environ.kaggle_key)
    print('Kaggle config dir', h_os_environ.kaggle_config_dir)

    sql_string = 'select ref, "lastUpdated" from dataset_control where ref = ' + "'" + dataset_ref + "'"
    with h_sqlalchemy.engine.connect() as conn:
        rs = conn.execute(sql_string)

    for row in rs:
        try:
            print('Azure Cord19 last updated:', str(row[1]))
            ti.xcom_push(key='AzureCord19lastUpdated', value=str(row[1]))
        except:
            pass


def fetch_kaggle_cord19_date(ti):
    print('--- Fetch Kaggle Cord19 last updated')
    h_kaggle = HelperKaggle()
    kaggle_dataset_info = h_kaggle.api.dataset_list(search='allen-institute-for-ai/CORD-19-research-challenge')
    for d in kaggle_dataset_info:
        if d.ref == dataset_ref:
            print('Kaggle Cord19 last updated:', str(d.lastUpdated))
            ti.xcom_push(key='KaggleCord19lastUpdated', value=str(d.lastUpdated))


def is_kaggle_cord19_newer_version(ti):
    print('--- Is Kaggle Cord19 newer version')
    azure_cord19_last_updated = ti.xcom_pull(key='AzureCord19lastUpdated', task_ids=['fetch_azure_cord19_date'])
    kaggle_cord19_last_updated = ti.xcom_pull(key='KaggleCord19lastUpdated', task_ids=['fetch_kaggle_cord19_date'])
    print('Azure date:', azure_cord19_last_updated)
    print('Kaggle date:', kaggle_cord19_last_updated)
    if kaggle_cord19_last_updated > azure_cord19_last_updated:
        print('Newer dataset version available')
        return 'extract_cord19_dataset'
    print('No newer dataset version available')
    return 'do_not_extract_cord19_dataset'


def extract_cord19_dataset():
    print('--- Extract Cord19 Dataset')
    h_kaggle = HelperKaggle()
    print('Before call to download')
    h_kaggle.api.dataset_download_files(dataset_ref, path='cord19', quiet=False, unzip=True)
    print('After call to download')
    path = os.getcwd()
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        print(f)
    print('CWD', path)


def prepare_cord19_upload():
    print('--- Prepare Cord19 Upload')
    shutil.rmtree('cord19/Kaggle')
    shutil.rmtree('cord19/cord_19_embeddings')


# azcopy remove "https://acieurfrcaasssthns.dfs.core.windows.net/datasets/test" --recursive;
def delete_old_azure_cord19_blob1():
    print('--- Delete old Azure cord19 blob')
    h_os_environ = HelperOsEnviron()
    container_client = ContainerClient.from_connection_string(conn_str=h_os_environ.az_rp_mlw_blob_connect_string,
                                                              container_name='datasets')
    container_client.delete_blob(blob='kaggle-cord19')


def upload_cord19_dataset1():
    print('--- Upload Cord19 Dataset')
    container_name = 'datasets'
    h_os_environ = HelperOsEnviron()
    blob_name = 'test'
    source = 'cord19'
    blob_service_client = BlobServiceClient.from_connection_string(h_os_environ.az_rp_mlw_blob_connect_string)
    prefix = '' if blob_name == '' else blob_name + '/'
    #prefix += os.path.basename(source) + '/'
    prefix += '/'
    for root, dirs, files in os.walk("cord19"):
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
                blob_client.upload_blob(data=data, blob_type="BlockBlob", overwrite=True)
                blob_client.upload_blob


def update_azure_cord19_date(ti):
    print('--- Update Azure Cord19 last updated')
    kaggle_cord19_last_updated = ti.xcom_pull(key='KaggleCord19lastUpdated', task_ids=['fetch_kaggle_cord19_date'])
    h_os_environ = HelperOsEnviron()
    h_sqlalchemy = HelperSqlalchemy(h_os_environ, 'analytics-shared-repository')

    sql_string = 'update dataset_control set "lastUpdated" = ' + '"' + kaggle_cord19_last_updated + '"' + \
                 ' where ref = ' + "'" + dataset_ref + "'"
    print(sql_string)
    with h_sqlalchemy.engine.connect() as conn:
        rs = conn.execute(sql_string)


def trigger_azureml_pipeline_cord19_dataprep():
    pass


def trigger_azureml_pipeline_cord19_model():
    pass


def finished():
    return None


start = DummyOperator(
    task_id='start'
)

fetch_azure_cord19_date = PythonOperator(
    task_id='fetch_azure_cord19_date',
    python_callable=fetch_azure_cord19_date,
    dag=dag)

fetch_kaggle_cord19_date = PythonOperator(
    task_id='fetch_kaggle_cord19_date',
    python_callable=fetch_kaggle_cord19_date,
    dag=dag)

is_kaggle_cord19_newer_version = BranchPythonOperator(
    task_id='is_kaggle_cord19_newer_version',
    python_callable=is_kaggle_cord19_newer_version,
    dag=dag,
)

do_not_extract_cord19_dataset = DummyOperator(
    task_id='do_not_extract_cord19_dataset'
)

extract_cord19_dataset = PythonOperator(
    task_id='extract_cord19_dataset',
    python_callable=extract_cord19_dataset,
    dag=dag)

prepare_cord19_upload = PythonOperator(
    task_id='prepare_cord19_upload',
    python_callable=prepare_cord19_upload,
    dag=dag)

#delete_old_azure_cord19_blob = PythonOperator(
#    task_id='delete_old_azure_cord19_blob',
#    python_callable=delete_old_azure_cord19_blob,
#    dag=dag)

upload_cord19_dataset = BashOperator(
    task_id='upload_cord19_dataset',
    bash_command='az storage blob upload-batch -d datasets/kaggle-cord19 -s /opt/airflow/cord19 --connection-string ' +
                 '"DefaultEndpointsProtocol=https;' +
                 'AccountName=xxx;' +
                 'AccountKey=xxx==;' +
                 'EndpointSuffix=core.windows.net"',
    dag=dag
)

# azcopy remove "https://acieurfrcaasssthns.dfs.core.windows.net/datasets/test" --recursive;
#def delete_old_azure_cord19_blob():
#    print('--- Delete old Azure cord19 blob')
#    h_os_environ = HelperOsEnviron()
#    container_client = ContainerClient.from_connection_string(conn_str=h_os_environ.az_rp_mlw_blob_connect_string,
#                                                              container_name='datasets')
#    container_client.delete_blob(blob='kaggle-cord19')


h_os_environ = HelperOsEnviron()
delete_old_azure_cord19_blob = BashOperator(
    task_id='delete_old_azure_cord19_blob',
    bash_command="/usr/bin/azcopy remove " +
                 '"https://xxxazurexxx.dfs.core.windows.net/datasets/kaggle-cord19?sv=2020-02-10&ss=bfqt&srt=co&'
                 'sp=xxx&'
                 'spr=xxxD"' 
                 " --recursive=true",
    dag=dag
)


update_azure_cord19_date = PythonOperator(
    task_id='update_azure_cord19_date',
    python_callable=update_azure_cord19_date,
    dag=dag)


#upload_cord19_dataset = PythonOperator(
#    task_id='upload_cord19_dataset',
#    python_callable=upload_cord19_dataset,
#    dag=dag)

trigger_azureml_pipeline_cord19_dataprep = PythonOperator(
    task_id='trigger_azureml_pipeline_cord19_dataprep',
    python_callable=trigger_azureml_pipeline_cord19_dataprep,
    dag=dag)

trigger_azureml_pipeline_cord19_model = PythonOperator(
    task_id='trigger_azureml_pipeline_cord19_model',
    python_callable=trigger_azureml_pipeline_cord19_model,
    dag=dag)

send_email_success = EmailOperator(
        task_id='send_email_success',
        to='jonpaulboyd@hotmail.co.uk',
        subject='Kaggle CORD-19 Dataset successfully updated',
        html_content=""" <h3>Heading for successfully updated</h3> """,
        dag=dag
)

finish = DummyOperator(
    task_id='finish',
    trigger_rule='none_failed_or_skipped'
)

start >> fetch_azure_cord19_date >> fetch_kaggle_cord19_date >> is_kaggle_cord19_newer_version >> \
[extract_cord19_dataset, do_not_extract_cord19_dataset]

do_not_extract_cord19_dataset >> finish

extract_cord19_dataset >> prepare_cord19_upload >> delete_old_azure_cord19_blob >> upload_cord19_dataset >> \
update_azure_cord19_date  >> trigger_azureml_pipeline_cord19_dataprep >> trigger_azureml_pipeline_cord19_model >> \
send_email_success >> finish


