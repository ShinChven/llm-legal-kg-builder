import psycopg2
from dotenv import load_dotenv

from src.db.db_connection import db_connection

load_dotenv()  # Safe to call multiple times; retained in case script run standalone.


def create_table():
    """Create the legislations table, indexes, and trigger using the pooled DB connection."""
    conn = db_connection.get_connection()
    if conn is None:
        print("Could not obtain a database connection from the pool.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS legislations (
                id SERIAL PRIMARY KEY,
                title VARCHAR(512),
                source VARCHAR(50),
                doc_type VARCHAR(50),
                version VARCHAR(50),
                act_id VARCHAR(50),
                in_amend BOOLEAN,
                ird_numbering VARCHAR(10),
                year INTEGER,
                act_no VARCHAR(10),
                act_type VARCHAR(50),
                date_as_at DATE,
                date_assent DATE,
                date_terminated DATE,
                date_first_valid DATE,
                stage VARCHAR(50),
                terminated VARCHAR(50),
                dlm VARCHAR(50),
                xml TEXT,
                text TEXT,
                word_count INTEGER,
                xml_link VARCHAR(2000) UNIQUE NOT NULL,
                pdf_link VARCHAR(2000),
                title_embedding REAL[],
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Indexes to speed up queries filtering by source or word_count
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_legislations_source
                ON legislations (source);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_legislations_word_count
                ON legislations (word_count);
        """)
        # Indexes to accelerate lookups by title / (year,title) used in relationship extraction
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_legislations_title
                ON legislations (title);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_legislations_year_title
                ON legislations (year, title);
        """)
        cursor.execute("""
            CREATE OR REPLACE FUNCTION set_updated_at()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        cursor.execute("""
            CREATE TRIGGER trigger_set_updated_at
            BEFORE UPDATE ON legislations
            FOR EACH ROW
            EXECUTE FUNCTION set_updated_at();
        """)
        conn.commit()
        print("Table 'legislations' and triggers created successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while creating table: {error}")
    finally:
        if conn:
            # Return connection to pool instead of closing underlying socket.
            from src.db.db_connection import db_connection as _dbc
            _dbc.release_connection(conn)

if __name__ == "__main__":
    create_table()
