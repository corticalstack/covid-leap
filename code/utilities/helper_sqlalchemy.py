from sqlalchemy import create_engine, exc


class HelperSqlalchemy:
    def __init__(self, os_environ, dbname):
        self.os_environ = os_environ
        self.db_conn_path = None
        self.engine = None

        self.set_db_conn_path(dbname)
        self.set_sqlalchemy_engine()

    def set_db_conn_path(self, dbname):
        self.db_conn_path = 'postgresql://' + self.os_environ.az_rp_psql_user + \
                            ':' + self.os_environ.az_rp_psql_pwd + \
                            '@' + self.os_environ.az_rp_psql_host + \
                            ':5432/' + dbname

    def set_sqlalchemy_engine(self):
        self.engine = create_engine(self.db_conn_path, connect_args={'sslmode': 'require'})

