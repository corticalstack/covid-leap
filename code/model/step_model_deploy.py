import argparse
import os
import re
import pandas as pd
import torch
from azureml.core import Run
from sentence_transformers import SentenceTransformer, models
from gensim.summarization.textcleaner import split_sentences
import joblib
from sqlalchemy import create_engine
from azureml.core import Run, Workspace, Model
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.exceptions import WebserviceException


class ModelDeploy:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.get_runtime_arguments()

        self.df = pd.DataFrame()
        self.dataset = None
        self.corpus_embeddings = None

        self.build_model = True
        self.path_corpus_as_df = 'cord19_bert_as_df.pkl'

        self.model = None
        self.corpus_embeddings = None

        self.register_model()

        if self.args.deploy:
            self.deploy_model()

    @staticmethod
    def set_workspace(svc_pr):
        ws = Workspace(
            subscription_id="xxx",
            resource_group="aci-eur-frc-aa-ss-rg",
            workspace_name="aci-eur-frc-aa-ss-mlw",
            auth=svc_pr)
        return ws

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--wrkdir',
            type=str,
            help='Model working directory'
        )

        parser.add_argument('--deploy', dest='deploy', action='store_true')
        parser.add_argument('--no-deploy', dest='deploy', action='store_false')
        parser.set_defaults(deploy=True)

        self.args = parser.parse_args()

        print('Working Directory: {}'.format(self.args.wrkdir))
        print('Deploy: {}'.format(self.args.deploy))

    def register_model(self):
        print('Register model')
        wrkdir_path_corpus_as_df = self.args.wrkdir + '/' + self.path_corpus_as_df
        self.path_corpus_as_df = 'outputs/' + self.path_corpus_as_df
        self.run.upload_file(name=self.path_corpus_as_df, path_or_stream=wrkdir_path_corpus_as_df)
        model = self.run.register_model(model_name='cord19_bert',
                                        tags={'type': 'bert'},
                                        model_path=self.path_corpus_as_df)
        print(model.name, model.id, model.version, sep='\t')

    def deploy_model(self):
        print('Deploy model')

        rp_env = self.configure_runtime_env()
        svc_pr = self.set_service_principal()
        ws = self.set_workspace(svc_pr)

        inference_config = InferenceConfig(entry_script='inference_cord19_search.py',
                                           environment=rp_env)

        # See this link for Azure Container Instances resource limits per region
        # https: // docs.microsoft.com / en - us / azure / container - instances / container - instances - region - availability
        #aci_config = AciWebservice.deploy_configuration(cpu_cores=2, memory_gb=8)

        # JP - option to use AKS with GPU if necessary
        # Example from here:
        # https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-inferencing-gpus

        from azureml.core.webservice import AksWebservice
        from azureml.core.compute import ComputeTarget, AksCompute
        from azureml.exceptions import ComputeTargetException

        # Choose a name for your cluster
        aks_name = "gpuinc6sv3"

        # Check to see if the cluster already exists
        try:
            aks_target = ComputeTarget(workspace=ws, name=aks_name)
            print('Found existing compute target')
        except ComputeTargetException:
            print('Creating a new compute target...')
            # Provision AKS cluster with GPU machine
            prov_config = AksCompute.provisioning_configuration(vm_size="Standard_NC6s_v3")

            # Create the cluster
            aks_target = ComputeTarget.create(
                workspace=ws, name=aks_name, provisioning_configuration=prov_config
            )

            aks_target.wait_for_completion(show_output=True)

        from azureml.core.webservice import AksWebservice

        # https://docs.microsoft.com/en-us/azure/machine-learning/how-to-authenticate-web-service
        # https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-deploy-azure-kubernetes-service.md

        gpu_aks_config = AksWebservice.deploy_configuration(autoscale_enabled=False,
                                                            num_replicas=2,
                                                            cpu_cores=4,
                                                            memory_gb=16,
                                                            auth_enabled=False)

        model = Model(ws, 'cord19_bert')
        service_name = 'cord19-bert-search-service'

        # delete service if it exists
        try:
            service = Webservice(ws, name=service_name)
            if service:
                service.delete()
                print('Deleted existing service')
        except WebserviceException as e:
            print('Error when deleting service: ', e)


        # Deploy the model
        aks_service = Model.deploy(ws,
                                   models=[model],
                                   inference_config=inference_config,
                                   deployment_config=gpu_aks_config,
                                   deployment_target=aks_target,
                                   name=service_name)

        #aks_service.wait_for_deployment(show_output=True)
        #print(aks_service.state)

        # service = Model.deploy(workspace=ws,
        #                        name=service_name,
        #                        models=[model],
        #                        inference_config=inference_config,
        #                        deployment_config=aci_config,
        #                        overwrite=True)
        #print(service.state)

    def configure_runtime_env(self):
        rp_env = Environment(name='rp')
        rp_env.docker.enabled = True
        rp_env.python.user_managed_dependencies = True
        rp_env.docker.base_image_registry.address = "acieurfrcaassacr.azurecr.io"
        rp_env.docker.base_image_registry.username = self.run.get_secret(name='acieurfrcaassacr-admin-user')
        rp_env.docker.base_image_registry.password = self.run.get_secret(name='acieurfrcaassacr-admin-pwd')
        rp_env.inferencing_stack_version = "latest"
        rp_env.docker.base_image = "acieurfrcaassacr.azurecr.io/azureml-env-base-research-platform:latest"
        return rp_env

    def set_service_principal(self):
        svc_key = self.run.get_secret(name='xxx')

        svc_pr = ServicePrincipalAuthentication(
            tenant_id='xxx',
            service_principal_id='xxx',
            service_principal_password=svc_key)

        return svc_pr


if __name__ == '__main__':
    print('--- Model deploy')
    model_deploy = ModelDeploy()
    print('--- Model deploy completed')
