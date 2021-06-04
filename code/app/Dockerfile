FROM python:3.7-slim

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

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501 3000 80
WORKDIR /app
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]





