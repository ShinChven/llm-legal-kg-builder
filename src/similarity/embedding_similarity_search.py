import numpy as np
from src.db.db_connection import db_connection


def find_similar_acts_by_embedding(
    query_embedding: list, threshold: float = 0.99, limit: int = 10
):
    """
    Finds similar acts based on a given embedding using pgvector.

    Args:
        query_embedding: The embedding of the act to search for.
        threshold: The similarity threshold to filter results.
        limit: The maximum number of similar acts to return.

    Returns:
        A list of similar acts, each represented as a dictionary.
    """
    conn = db_connection.get_connection()
    if not conn:
        print("Could not obtain a database connection.")
        return []

    results = []
    try:
        with conn.cursor() as cursor:
            # The <=> operator in pgvector calculates cosine distance (1 - cosine_similarity)
            # So, we filter by distance <= (1 - threshold)
            # We also select 1 - (title_embedding <=> %s) to get the cosine similarity
            embedding_list = (
                query_embedding.tolist()
                if isinstance(query_embedding, np.ndarray)
                else query_embedding
            )
            cursor.execute(
                """SELECT title, 1 - (title_embedding::vector <=> %s::vector) AS similarity
                   FROM legislations
                   WHERE 1 - (title_embedding::vector <=> %s::vector) >= %s
                   ORDER BY similarity DESC
                   LIMIT %s""",
                (embedding_list, embedding_list, threshold, limit),
            )
            all_acts = cursor.fetchall()

            for act_title, similarity in all_acts:
                results.append({"title": act_title, "similarity": similarity})

    except Exception as e:
        print(f"An error occurred during similarity search: {e}")
    finally:
        if conn:
            db_connection.release_connection(conn)

    return results
