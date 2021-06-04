import os


class HelperOsEnviron:
    def __init__(self):
        self.az_rp_psql_host = None
        self.az_rp_psql_user = None
        self.az_rp_psql_pwd = None

        self.az_rp_mlw_blob_connect_string = None

        self.kaggle_username = None
        self.kaggle_key = None
        self.kaggle_config_dir = None

        self.get_os_environ_variables()

    def get_os_environ_variables(self):
        self.az_rp_psql_host = os.environ.get('AZ_RP_PSQL_HOST')
        self.az_rp_psql_user = os.environ.get('AZ_RP_PSQL_USER')
        self.az_rp_psql_pwd = os.environ.get('AZ_RP_PSQL_PWD')

        self.az_rp_mlw_blob_connect_string = os.environ.get('AZ_RP_MLW_BLOB_CONNECT_STRING')

        self.kaggle_username = os.environ.get('KAGGLE_USERNAME')
        self.kaggle_key = os.environ.get('KAGGLE_KEY')
        self.kaggle_config_dir = os.environ.get('KAGGLE_CONFIG_DIR')


