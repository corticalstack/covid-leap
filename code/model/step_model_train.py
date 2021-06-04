import argparse
import os
import re
import pandas as pd
from azureml.core import Run, Workspace
from azureml.core.model import Model
from azure.storage.blob import BlobServiceClient
from azureml.core.authentication import ServicePrincipalAuthentication
from sentence_transformers import SentenceTransformer, InputExample, losses, models
import torch
from torch.utils.data import DataLoader
from datetime import datetime


class ModelTrain:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.get_runtime_arguments()

        if torch.cuda.is_available():
            print('Cuda is available')
        else:
            print('Cuda is unavailable')

        self.blob_connect_string = None
        self.get_os_environ_variables()

        torch.cuda.empty_cache()

        self.train_examples = []
        self.df = pd.DataFrame()

        self.model = None
        self.model_path = None

        self.load_generated_queries()
        self.input_example_training_queries()

        self.train_model()
        self.register_model()

    @staticmethod
    def clean_sentences(sent):
        sent = re.sub(r'http\S+', 'URL', sent, flags=re.MULTILINE)  # Remove URLs
        sent = re.sub(r'[^\w\s]', '', sent)  # Remove punctuation
        if sent:
            return sent

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--base_model_name',
            type=str,
            help='Model name'
        )
        parser.add_argument(
            '--register_name',
            type=str,
            help='Register name'
        )
        parser.add_argument(
            '--traindir',
            type=str,
            help='Model training directory'
        )
        self.args = parser.parse_args()

        print('Base model name: {}'.format(self.args.base_model_name))
        print('Register name: {}'.format(self.args.register_name))
        print('Training Directory: {}'.format(self.args.traindir))

    def get_os_environ_variables(self):
        print('--- Getting OS Environment Variables')
        self.blob_connect_string = os.environ.get('AZ_RP_MLW_BLOB_CONNECT_STRING')

    def load_generated_queries(self):
        print('--- Loading generated queries')

        container_name = 'modelling'
        blob_name = 'cord19_generated_queries_for_model_training.csv'
        blob_service_client = BlobServiceClient.from_connection_string(self.blob_connect_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        with open(blob_name, 'wb') as gq_blob:
            download_stream = blob_client.download_blob()
            gq_blob.write(download_stream.readall())

        colnames = ['query', 'paragraph']
        self.df = pd.read_csv(blob_name, delimiter='\t', names=colnames, header=None,
                              dtype={'query': str, 'paragraph': str})
        print('Qery paragraph dataset size: ', str(len(self.df)))

    def input_example_training_queries(self):
        print('--- Input example training queries')
        for index, row in self.df.iterrows():
            self.train_examples.append(InputExample(texts=[row['query'], row['paragraph']]))

    def train_model(self):
        print('--- Train model')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Training started:', current_time)

        train_dataloader = DataLoader(self.train_examples, batch_size=16)

        print('Creating model')

        # Create a SentenceTransformer model from scratch
        word_emb = models.Transformer(self.args.base_model_name)
        pooling = models.Pooling(word_emb.get_word_embedding_dimension())

        print('Building model')
        self.model = SentenceTransformer(modules=[word_emb, pooling])

        print('Losses')
        # MultipleNegativesRankingLoss requires input pairs (query, relevant_passage)
        # and trains the model so that is is suitable for semantic search
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        print('Tuning model')
        num_epochs = 1
        print('Number epochs: ', str(num_epochs))
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
        print('Warmup steps: ', str(warmup_steps))
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Training completed:', current_time)

        self.model_path = self.args.traindir + '/test_model'
        #self.model_path = 'c19gq_model'
        self.model.save(self.model_path)

    def register_model(self):
        #svc_key = self.run.get_secret(name='aci-eur-frc-aa-ss-mlw-svc-principal-key')
        svc_key = 'xxx'
        self.model_path = 'c19gq_model'
        svc_pr = ServicePrincipalAuthentication(
            tenant_id='xxx',
            service_principal_id='xxx',
            service_principal_password=svc_key)

        ws = Workspace(
            subscription_id="xxx",
            resource_group="xxx",
            workspace_name="xxx",
            auth=svc_pr
        )

        model = Model.register(workspace=ws,
                               model_path=self.model_path,
                               model_name=self.args.register_name,
                               tags={'dataset': 'cord-19', 'base model': self.args.base_model_name},
                               description='Covid-19 cord19 corpus generated query trained model'
                               )


if __name__ == '__main__':
    print('--- Model train started')
    model_train = ModelTrain()
    print('--- Model train  completed')
