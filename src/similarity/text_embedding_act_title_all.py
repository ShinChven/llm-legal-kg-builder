import concurrent.futures
from src.db.db_connection import db_connection
from src.similarity.text_embedding_act_title import get_and_update_act_embeddings

# The maximum number of concurrent threads to use.
MAX_WORKERS = 1
BATCH_SIZE = 5  # As recommended by Google API docs for gemini-embedding-001


def get_unprocessed_legislations():
    """Fetches all legislation titles where title_embedding is NULL."""
    conn = db_connection.get_connection()
    if conn is None:
        print("Could not obtain a database connection.")
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT title FROM legislations WHERE title_embedding IS NULL AND title IS NOT NULL"
            )
            titles = [row[0] for row in cursor.fetchall()]
            return titles
    except Exception as e:
        print(f"Database error while fetching titles: {e}")
        return []
    finally:
        db_connection.release_connection(conn)


def main():
    """Main function to fetch and process all unprocessed legislations in batches."""
    titles_to_process = get_unprocessed_legislations()
    if not titles_to_process:
        print("No legislations to process.")
        return

    print(f"Found {len(titles_to_process)} legislations to process.")

    # Create batches of titles
    title_batches = [
        titles_to_process[i : i + BATCH_SIZE]
        for i in range(0, len(titles_to_process), BATCH_SIZE)
    ]
    print(f"Processing in {len(title_batches)} batches of size up to {BATCH_SIZE}.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(get_and_update_act_embeddings, batch): batch
            for batch in title_batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                result = future.result()
                if not result or len(result) != len(batch):
                    print(f"Failed to process batch starting with: {batch[0]}")
            except Exception as exc:
                print(f"Batch starting with {batch[0]} generated an exception: {exc}")


if __name__ == "__main__":
    main()
    db_connection.close_all_connections()
