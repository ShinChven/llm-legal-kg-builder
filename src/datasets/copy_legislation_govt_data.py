import psycopg2
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

class ExtraDBConnection:
    """Database connection for the extra database containing legislation.govt.nz data"""

    def __init__(self):
        self.connection = None

    def connect(self):
        """Connect to the extra database"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                database=os.getenv("EXTRA_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD")
            )
            print("Connected to extra database successfully.")
            return self.connection
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error connecting to extra database: {error}")
            return None

    def disconnect(self):
        """Close the connection to the extra database"""
        if self.connection:
            self.connection.close()
            print("Extra database connection closed.")

class MainDBConnection:
    """Database connection for the main database"""

    def __init__(self):
        self.connection = None

    def connect(self):
        """Connect to the main database"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                database=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD")
            )
            print("Connected to main database successfully.")
            return self.connection
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error connecting to main database: {error}")
            return None

    def disconnect(self):
        """Close the connection to the main database"""
        if self.connection:
            self.connection.close()
            print("Main database connection closed.")

def fetch_legislation_govt_data(extra_db_conn) -> List[Dict[str, Any]]:
    """Fetch all legislation.govt.nz data from the extra database"""
    cursor = extra_db_conn.cursor()

    query = """
        SELECT *
        FROM legislation_documents
        WHERE source = 'legislation.govt.nz'
    """

    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()

        # Convert to list of dictionaries
        data = []
        for row in results:
            data.append(dict(zip(columns, row)))

        print(f"Found {len(data)} records from legislation.govt.nz in extra database")
        return data

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error fetching legislation.govt.nz data: {error}")
        return []
    finally:
        cursor.close()

def check_title_exists(main_db_conn, title: str) -> bool:
    """Check if a title already exists in the main database"""
    cursor = main_db_conn.cursor()

    query = "SELECT COUNT(*) FROM legislations WHERE title = %s"

    try:
        cursor.execute(query, (title,))
        count = cursor.fetchone()[0]
        return count > 0

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error checking if title exists: {error}")
        return True  # Assume exists to avoid duplicates on error
    finally:
        cursor.close()

def insert_record(main_db_conn, record: Dict[str, Any]) -> bool:
    """Insert a single record into the main database"""
    cursor = main_db_conn.cursor()

    # Map fields from extra database to main database
    insert_query = """
        INSERT INTO legislations (
            title, source, doc_type, year, act_no, text, word_count, pdf_link, xml_link, act_type, xml, version, dlm
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """

    # Map the fields according to requirements
    mapped_data = (
        record['title'],                    # title -> title
        'www.legislation.govt.nz',             # source -> 'legislation.govt.nz'
        'act',                             # doc_type -> 'act'
        record['year'],                    # year -> year
        record['number_of_legislation'],   # number_of_legislation -> act_no
        record['text'],                    # text -> text
        record['word_count'],              # word_count -> word_count
        record['pdf_link'],                # pdf_link -> pdf_link
        record['link'],                    # link -> xml_link
        record['issuer'],                  # issuer -> act_type
        record['html'],                    # html -> xml
        record['version'],                 # version -> version
        record['dlm']                      # dlm -> dlm
    )

    try:
        cursor.execute(insert_query, mapped_data)
        main_db_conn.commit()
        return True

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error inserting record '{record['title']}': {error}")
        main_db_conn.rollback()
        return False
    finally:
        cursor.close()

def copy_legislation_govt_data():
    """Main function to copy legislation.govt.nz data from extra database to main database"""
    extra_db = ExtraDBConnection()
    main_db = MainDBConnection()

    extra_conn = extra_db.connect()
    main_conn = main_db.connect()

    if not extra_conn or not main_conn:
        print("Failed to establish database connections")
        return

    try:
        # Fetch legislation.govt.nz data from extra database
        legislation_records = fetch_legislation_govt_data(extra_conn)

        if not legislation_records:
            print("No legislation.govt.nz records found to copy")
            return

        inserted_count = 0
        skipped_count = 0
        error_count = 0

        # Process each record
        for record in legislation_records:
            title = record['title']

            if not title:
                print(f"Skipping record with empty title")
                skipped_count += 1
                continue

            # Check if title already exists in main database
            if check_title_exists(main_conn, title):
                print(f"Skipping existing title: {title[:50]}...")
                skipped_count += 1
                continue

            # Insert the record
            if insert_record(main_conn, record):
                print(f"Inserted: {title[:50]}...")
                inserted_count += 1
            else:
                print(f"Failed to insert: {title[:50]}...")
                error_count += 1

        print(f"\nCopy operation completed:")
        print(f"  Records inserted: {inserted_count}")
        print(f"  Records skipped (already exist): {skipped_count}")
        print(f"  Records with errors: {error_count}")
        print(f"  Total records processed: {len(legislation_records)}")

    except Exception as e:
        print(f"Unexpected error during copy operation: {e}")

    finally:
        # Close connections
        extra_db.disconnect()
        main_db.disconnect()

if __name__ == "__main__":
    print("Starting legislation.govt.nz data copy operation...")
    copy_legislation_govt_data()
    print("Copy operation finished.")
