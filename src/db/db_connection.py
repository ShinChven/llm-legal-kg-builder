import psycopg2
from psycopg2 import pool
import os
from dotenv import load_dotenv

load_dotenv()

class DBConnection:
    _instance = None
    _pool = None
    _minconn = 1
    _maxconn = 10

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBConnection, cls).__new__(cls)
            try:
                # Configurable threaded pool for multi-worker support
                cls._minconn = int(os.getenv("POSTGRES_POOL_MIN", "1"))
                cls._maxconn = int(os.getenv("POSTGRES_POOL_MAX", "10"))
                cls._pool = psycopg2.pool.ThreadedConnectionPool(
                    cls._minconn,
                    cls._maxconn,
                    host=os.getenv("POSTGRES_HOST"),
                    port=os.getenv("POSTGRES_PORT"),
                    database=os.getenv("POSTGRES_DB"),
                    user=os.getenv("POSTGRES_USER"),
                    password=os.getenv("POSTGRES_PASSWORD")
                )
                print(
                    f"Database threaded connection pool created successfully. "
                    f"min={cls._minconn} max={cls._maxconn}"
                )
            except (Exception, psycopg2.DatabaseError) as error:
                print(f"Error while creating connection pool: {error}")
                cls._instance = None
        return cls._instance

    def get_connection(self):
        if self._pool:
            return self._pool.getconn()
        return None

    def release_connection(self, conn):
        if self._pool:
            self._pool.putconn(conn)

    def close_all_connections(self):
        if self._pool:
            self._pool.closeall()
            print("All database connections are closed.")

    def get_pool_limits(self):
        """Return (minconn, maxconn) configured for the pool."""
        return (self._minconn, self._maxconn)

db_connection = DBConnection()
