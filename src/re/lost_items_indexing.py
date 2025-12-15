import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.db.db_connection import db_connection
from src.similarity.text_embedding_act_title import get_embeddings
from src.db.create_lost_and_found_table import ensure_lost_and_found_table

def get_lost_acts():
    """Fetches object_names from act_relationships that are not in legislations."""
    conn = db_connection.get_connection()
    if not conn:
        print("Could not obtain a database connection.")
        return []

    lost_acts = []
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT ar.object_name
                FROM act_relationships ar
                LEFT JOIN legislations l ON ar.object_name = l.title
                WHERE l.title IS NULL;
                """
            )
            lost_acts = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"An error occurred while fetching lost acts: {e}")
    finally:
        if conn:
            db_connection.release_connection(conn)
    return lost_acts

def store_lost_acts_with_embeddings(lost_acts):
    """Generates embeddings for lost acts and stores them in the lost_and_found table."""
    if not lost_acts:
        print("No lost acts to process.")
        return

    batch_size = 100
    for i in range(0, len(lost_acts), batch_size):
        batch = lost_acts[i:i + batch_size]
        embeddings_dict = get_embeddings(batch)
        if not embeddings_dict:
            print(f"Could not generate embeddings for batch starting at index {i}.")
            continue

        conn = db_connection.get_connection()
        if not conn:
            print("Could not obtain a database connection.")
            # No need to return, as we want to continue with the next batch
            continue

        try:
            with conn.cursor() as cursor:
                update_data = [
                    (title, embedding)
                    for title, embedding in embeddings_dict.items()
                ]
                cursor.executemany(
                    """INSERT INTO lost_and_found (object_name, title_embedding)
                       VALUES (%s, %s)
                       ON CONFLICT (object_name) DO NOTHING;""",
                    update_data,
                )
                conn.commit()
                print(f"Successfully processed and stored batch of {len(update_data)} lost acts.")

        except Exception as e:
            print(f"An error occurred while storing lost acts with embeddings: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                db_connection.release_connection(conn)

if __name__ == "__main__":
    print("Ensuring 'lost_and_found' table exists...")
    ensure_lost_and_found_table()
    print("Fetching lost acts...")
    lost_acts = get_lost_acts()
    if lost_acts:
        print(f"Found {len(lost_acts)} lost acts.")
        store_lost_acts_with_embeddings(lost_acts)
    else:
        print("No lost acts found.")
    db_connection.close_all_connections()
