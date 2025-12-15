import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.db.db_connection import db_connection

load_dotenv()

# Initialize the client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def get_embeddings(texts: list[str]) -> dict[str, list[float]]:
    """
    Generates embeddings for a given list of texts using Google's gemini-embedding-001 model.

    Args:
        texts: A list of texts to embed.

    Returns:
        A dictionary where keys are the input texts and values are their embeddings.
    """
    if not texts:
        return {}

    try:
        embedding_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        ).embeddings

        return {text: emb.values for text, emb in zip(texts, embedding_result)}
    except Exception as e:
        print(f"An error occurred while generating embeddings: {e}")
        return {}


def get_and_update_act_embeddings(act_titles: list[str]) -> dict[str, list[float]]:
    """
    For a list of act titles, gets their embeddings, updates the database,
    and returns the embeddings.

    Args:
        act_titles: A list of act titles to process.

    Returns:
        A dictionary mapping act titles to their embeddings.
    """
    if not act_titles:
        return {}

    conn = db_connection.get_connection()
    if not conn:
        print("Could not obtain a database connection.")
        return {}

    embeddings_dict = {}
    try:
        # Generate embeddings for all titles in one batch
        embeddings_dict = get_embeddings(act_titles)
        if not embeddings_dict:
            print("Could not generate embeddings for the given titles.")
            return {}

        # Update database with new embeddings
        with conn.cursor() as cursor:
            update_data = [
                (embedding, title)
                for title, embedding in embeddings_dict.items()
            ]
            cursor.executemany(
                "UPDATE legislations SET title_embedding = %s WHERE title = %s",
                update_data,
            )
            conn.commit()
            print(f"Successfully updated embeddings for {len(update_data)} titles.")

    except Exception as e:
        print(f"An error occurred: {e}")
        if conn:
            conn.rollback()
        return {}  # Return empty dict on error
    finally:
        if conn:
            db_connection.release_connection(conn)

    return embeddings_dict
