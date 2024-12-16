'''
Copyright (C) 2023 TuringVision

MysqlClient
'''

__all__ = ['MysqlClient']

class MysqlClient:

    TIMEOUT = 3.0
    def __init__(self, user='', password='', host='127.0.0.1', port=3306, database=''):
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._cnx = None
        self._connect()

    def _connect(self):
        import mysql.connector
        self._cnx = mysql.connector.connect(user=self._user, password=self._password,
                                            host=self._host, port=self._port,
                                            database=self._database)
    def close(self):
        if self._cnx:
            self._cnx.close()
            self._cnx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def query(self, cmds):
        cursor = self._cnx.cursor()
        cursor.execute(cmds)
        data = [d for d in cursor]
        column_names = cursor.column_names
        cursor.close()
        return column_names, data
