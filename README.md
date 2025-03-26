# 🔬 COVID-LEAP: COVID-19 Literature Exploration and Analysis Platform

A comprehensive platform for exploring COVID-19 research literature and vaccine development data.

## Description

COVID-LEAP (Literature Exploration and Analysis Platform) is a research tool that combines multiple COVID-19 data sources to provide researchers and healthcare professionals with an integrated view of the pandemic's scientific landscape. The platform processes the CORD-19 (COVID-19 Open Research Dataset) and clinical trials data to enable semantic search, visualization, and analysis of research papers and vaccine development progress.

## 🌟 Features

- **Interactive Research Dashboard**: Streamlit-based web application with multiple views
- **Semantic Search**: Advanced search capabilities for COVID-19 research papers with multiple ranking methods
- **Vaccine Development Tracking**: Visualization of clinical and preclinical vaccine candidates
- **Data Pipeline**: Automated ETL processes for keeping data current
- **Machine Learning**: NLP models for semantic search and document ranking
- **Containerized Deployment**: Docker-based deployment for both local and cloud environments

## 🔧 Prerequisites

- Python 3.7+
- Docker and Docker Compose
- Azure subscription (for cloud deployment)
- PostgreSQL database
- Elasticsearch instance
- Azure ML workspace (for model training and deployment)

## 🚀 Setup Guide

### Local Development

1. Clone the repository
2. Set up environment variables:
   ```
   AZ_RP_MLW_BLOB_CONNECT_STRING=<Azure Blob Storage connection string>
   AZ_RP_MLW_SVC_PRINCIPAL_KEY=<Azure Service Principal key>
   AZ_RP_PSQL_HOST=<PostgreSQL host>
   AZ_RP_PSQL_USER=<PostgreSQL user>
   AZ_RP_PSQL_PWD=<PostgreSQL password>
   AZ_RP_ES_HOST=<Elasticsearch host>
   AZ_RP_ES_USER=<Elasticsearch user>
   AZ_RP_ES_PWD=<Elasticsearch password>
   ```
3. Install dependencies:
   ```bash
   cd code/app
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   cd code/app
   streamlit run main.py
   ```

### Docker Deployment

1. Build and run using Docker Compose:
   ```bash
   cd code/app
   docker-compose up -d
   ```

### Azure Deployment

The repository includes configurations for deploying components to Azure:

- Azure ML for model training and deployment
- Azure Container Instances for application hosting
- Azure Kubernetes Service for high-performance model serving

## 📊 Architecture

The platform consists of several components:

1. **Data Preparation Pipeline**:
   - ETL processes for CORD-19 dataset, using an Azure Machine Learning pipeline
   - Clinical trials data processing
   - Feature engineering and topic modeling

2. **Model Training and Deployment**:
   - BERT-based embeddings for semantic search
   - BM25 text ranking
   - Cross-encoder reranking

3. **Web Application**:
   - Streamlit-based interactive dashboard
   - Multiple views for different aspects of COVID-19 research
   - Integration with search backend

## 📚 Data Sources

- **CORD-19**: COVID-19 Open Research Dataset from Allen Institute for AI
- **Clinical Trials**: Data from ClinicalTrials.gov and AACT database
- **WHO Vaccine Landscape**: Vaccine development tracking from World Health Organization

## 🔍 Usage

### Vaccine Overview

Explore the distribution of vaccine candidates by platform/type, with separate views for clinical and preclinical candidates.

### Clinical Candidates

Analyze clinical-stage vaccine candidates with detailed information about development phases, trial IDs, and other characteristics.

### Article Search

Search the CORD-19 dataset using different search methods:
- BM25 (keyword-based)
- Semantic search (embedding-based)
- BM25 + Semantic Rerank
- Semantic Cross-Encoder

Filter results by:
- Publication year
- Journal
- Topic
- Virus constraint

Apply paper metrics to enhance relevance:
- Citation count
- Author citation ratio
- PageRank
- Paper recency

## 🛠️ Development

### Project Structure

- **code/app**: Streamlit web application
- **code/dataprep**: Data preparation pipelines
  - **cord19**: CORD-19 dataset processing
  - **clinical-trials**: Clinical trials data processing
- **code/etl**: ETL processes for data ingestion
- **code/model**: Model training and deployment
- **code/operationalisation**: Airflow DAGs for workflow orchestration
- **code/utilities**: Helper functions and utilities
- **docker**: Docker configurations for different environments

## 📝 License

This project uses the CORD-19 dataset which is licensed under the Creative Commons Attribution License.

## 🔗 Resources

- [CORD-19 Dataset](https://www.semanticscholar.org/cord19)
- [ClinicalTrials.gov](https://clinicaltrials.gov/)
- [WHO COVID-19 Vaccine Landscape](https://www.who.int/publications/m/item/draft-landscape-of-covid-19-candidate-vaccines)
