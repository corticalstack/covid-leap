FROM apache/airflow


ARG az_rp_mlw_blob_connect_string
ARG az_rp_mlw_svc_principal_key
ARG az_rp_psql_host
ARG az_rp_psql_user
ARG az_rp_psql_pwd
ARG az_rp_es_host
ARG az_rp_es_user
ARG az_rp_es_pwd
ARG kaggle_username
ARG kaggle_key


ENV AZ_RP_MLW_BLOB_CONNECT_STRING=$az_rp_mlw_blob_connect_string
ENV AZ_RP_MLW_SVC_PRINCIPAL_KEY=$az_rp_mlw_svc_principal_key
ENV AZ_RP_PSQL_HOST=$az_rp_psql_host
ENV AZ_RP_PSQL_USER=$az_rp_psql_user
ENV AZ_RP_PSQL_PWD=$az_rp_psql_pwd
ENV AZ_RP_ES_HOST=$az_rp_es_host
ENV AZ_RP_ES_USER=$az_rp_es_user
ENV AZ_RP_ES_PWD=$az_rp_es_pwd
ENV KAGGLE_USERNAME=$kaggle_username
ENV KAGGLE_KEY=$kaggle_key
ENV KAGGLE_CONFIG_DIR='~/airflowenv/.kaggle'
ENV AZCOPY_LOG_LOCATION='/usr/bin/azcopylogs'

ENV SENDGRID_MAIL_FROM=corticalstackbot@gmail.com
ENV SENDGRID_API_KEY='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'

USER root

RUN apt-get update --fix-missing

RUN pip install sqlalchemy

RUN pip install kaggle

RUN pip install azure.storage.blob

# Install Azure CLI
RUN apt-get install ca-certificates curl apt-transport-https lsb-release gnupg -y &&\
    curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null &&\
    AZ_REPO=$(lsb_release -cs) &&\
    echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | sudo tee /etc/apt/sources.list.d/azure-cli.list &&\
    apt-get update &&\
    apt-get install azure-cli


RUN curl -L -o azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
RUN tar -xvf azcopy.tar.gz
RUN cp ./azcopy*/azcopy /usr/bin/
RUN rm -f azcopy.tar.gz
RUN chmod a+x /usr/bin/azcopy
RUN mkdir /usr/bin/azcopylogs
RUN chmod 777 /usr/bin/azcopylogs
RUN chmod 777 /opt/az/bin/az
RUN chmod 777 /home/airflow

RUN apt-get install nano

RUN pip install 'apache-airflow[sendgrid]'

COPY airflow.cfg airflow.cfg




