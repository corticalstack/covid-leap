FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04

ARG AZUREML_SDK_VERSION=1.19.0
ARG INFERENCE_SCHEMA_VERSION=1.1.0
ARG az_rp_mlw_blob_connect_string
ARG az_rp_mlw_svc_principal_key
ARG az_rp_psql_host
ARG az_rp_psql_user
ARG az_rp_psql_pwd
ARG az_rp_es_host
ARG az_rp_es_user
ARG az_rp_es_pwd


ENV AZ_RP_MLW_BLOB_CONNECT_STRING=$az_rp_mlw_blob_connect_string
ENV AZ_RP_MLW_SVC_PRINCIPAL_KEY=$az_rp_mlw_svc_principal_key
ENV AZ_RP_PSQL_HOST=$az_rp_psql_host
ENV AZ_RP_PSQL_USER=$az_rp_psql_user
ENV AZ_RP_PSQL_PWD=$az_rp_psql_pwd
ENV AZ_RP_ES_HOST=$az_rp_es_host
ENV AZ_RP_ES_USER=$az_rp_es_user
ENV AZ_RP_ES_PWD=$az_rp_es_pwd

RUN conda update --all --yes

RUN apt-get update --fix-missing

RUN conda install pytorch=1.6 torchvision cudatoolkit=10.1 -c pytorch

# pip version explicitly set as issues with ruamel package
# https://github.com/Azure/MachineLearningNotebooks/issues/1110
#RUN pip install --upgrade pip==20.1.1 --ignore-installed certifi

RUN pip install azureml-defaults==${AZUREML_SDK_VERSION} inference-schema==${INFERENCE_SCHEMA_VERSION} &&\
    pip install azureml-pipeline

RUN pip install azure-storage-blob

RUN pip install scipy==1.4.1 &&\
    pip install scispacy==0.2.5 &&\
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz &&\
    pip install spacy-langdetect &&\
    pip install nltk &&\
#    pip install transformers &&\
#    pip install sentence-transformers &&\
    pip install gensim

RUN pip install selenium &&\
    pip install lxml &&\
    pip install 'ipython-beautifulsoup[bs4]'

RUN pip install wordcloud &&\
    pip install matplotlib --ignore-installed &&\
    pip install plotly &&\
    pip install pyLDAvis &&\
    pip install ruamel-yaml &&\
    pip install networkx &&\
    pip install seaborn &&\
    pip install plotly

RUN apt-get install libpq-dev -y &&\
    pip install psycopg2 &&\
    pip install sqlalchemy

# Overcomes issue with import of cv2
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install python3-tk -y &&\
    pip install opencv-python &&\
    pip install camelot-py[cv] &&\
    conda install -c conda-forge ghostscript -y &&\
    conda install -c conda-forge faiss

RUN pip install xlrd

# Set numpy version to workaround pyarrow.lib.ArrowTypeError introduced by numpy 1.20
# ('Did not pass numpy.dtype object', 'Conversion failed for column X with type int64')
#RUN pip install numpy==1.19.5

RUN pip install elasticsearch

RUN pip install "modin[ray]"

#RUN pip install numpy --upgrade

#RUN conda install -c pytorch pytorch

#RUN conda install -c conda-forge pytorch
#RUN conda install -c conda-forge cudatoolkit=10.2
#RUN conda install -c pytorch pytorch==1.7.1
#RUN conda config --set channel_priority false
#RUN conda install pytorch=1.8 torchvision cudatoolkit=10.1 -c pytorch
#RUN conda install pytorch==1.6.0 cudatoolkit=10.1 -c conda-forge

#RUN pip install torch torchvision torchaudio
#RUN pip install torch===1.5.0 torchvision===0.6.0



RUN pip install sentencepiece
#RUN conda install -c huggingface transformers
RUN conda install -c huggingface tokenizers=0.9.4 transformers=4.2.2

RUN pip install sentence-transformers