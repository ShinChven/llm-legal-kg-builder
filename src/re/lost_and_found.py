
import numpy as np
from src.db.db_connection import db_connection
from src.similarity.embedding_similarity_search import find_similar_acts_by_embedding


def find_and_save_similar_acts(threshold: float = 0.8):
    """
    Finds similar acts for titles in the lost_and_found table and saves the results to a CSV file.

    Args:
        threshold: The similarity threshold to filter results.
    """
    conn = db_connection.get_connection()
    if not conn:
        print("Could not obtain a database connection.")
        return

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, object_name, title_embedding FROM lost_and_found where found_title is null;")
            lost_items = cursor.fetchall()

            for id, object_name, title_embedding in lost_items:
                print(f"Processing: {object_name}")
                if title_embedding:
                    similar_acts = find_similar_acts_by_embedding(
                        title_embedding, threshold, limit=1
                    )
                    if similar_acts:
                        top_match = similar_acts[0]
                        print(
                            f"  -> Found: {top_match['title']} (Similarity: {top_match['similarity']:.4f})"
                        )
                        update_query = "UPDATE lost_and_found SET found_title = %s, similarity = %s WHERE id = %s;"
                        cursor.execute(
                            update_query,
                            (top_match["title"], top_match["similarity"], id),
                        )
                        conn.commit()
                    else:
                        print(f"  -> No similar acts found for {object_name}")
                else:
                    print(f"  -> No embedding found for {object_name}, skipping.")

    except Exception as e:
        print(f"An error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_connection.release_connection(conn)


if __name__ == "__main__":
    find_and_save_similar_acts()
