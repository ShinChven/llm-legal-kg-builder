import os
import sys
import logging
from datetime import datetime
import concurrent.futures

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.re.extractor_llm import RelationshipExtractor
from src.re.act_relationship_handler import ActRelationshipHandler
from src.db.db_connection import db_connection

# --- CONFIGURATION ---
USE_CACHE = True  # Set to False to skip database cache and always run LLM
EXTRACTION_LAYERS = 1
MODEL_NAME = 'gemini-2.5-flash'
CHUNK_SIZE = 1000000
MAX_WORKERS_PER_ACT = 1   # Workers for processing chunks of a single act
MAX_CONCURRENT_ACTS = 10   # Number of acts to process in parallel
MIN_WORD_COUNT = 0      # Minimum word count for LLM processing
MAX_WORD_COUNT = 1000000    # Maximum word count for LLM processing

def get_unprocessed_acts():
    """
    Fetches a list of legislations (Acts) that have no recorded relationships.
    It checks for Acts in the 'legislations' table that do not appear in the 'act_relationships' table,
    either as a subject or an object.
    """
    print("Fetching acts with no relationships from the database...")
    conn = db_connection.get_connection()
    if not conn:
        print("Could not get a database connection.")
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    l.title
                FROM
                    legislations l
                LEFT JOIN
                    act_relationships ar ON l.title = ar.subject_name OR l.title = ar.object_name
                WHERE
                    l.word_count IS NOT NULL
                GROUP BY
                    l.title, l.year, l.word_count, l.pdf_link
                HAVING
                    COUNT(ar.id) = 0
                ORDER BY
                    l.word_count DESC;
            """)
            results = cursor.fetchall()
            
            if not results:
                print("No acts found without relationships.")
                return []

            # The query returns tuples, so we extract the first element (the title)
            unprocessed_acts = [row[0] for row in results]
            
            print(f"Found {len(unprocessed_acts)} acts to be processed.")
            return unprocessed_acts
            
    except Exception as e:
        print(f"Error fetching unprocessed acts: {e}")
        return []
    finally:
        db_connection.release_connection(conn)

def process_act(act_title, index):
    """
    Runs the relationship extraction process for a single act title.
    """
    logging.info(f"Starting extraction for act {index}: {act_title}")
    try:
        act_handler = ActRelationshipHandler()
        extractor = RelationshipExtractor(
            model_name=MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            max_workers=MAX_WORKERS_PER_ACT,
            act_handler=act_handler,
            use_cache=USE_CACHE
        )

        extractor.run_recursive(act_title, EXTRACTION_LAYERS)

        statistics = extractor.get_statistics()
        logging.info(f"Finished extraction for act {index}: {act_title}. Stats: {statistics}")
        return {act_title: statistics}

    except Exception as e:
        logging.error(f"A critical error occurred while processing act {index}: {act_title}: {e}", exc_info=True)
        return {act_title: {"error": str(e)}}

def main():
    """
    Main function to run the graph builder.
    """
    start_time = datetime.now()
    print("Starting relationship graph builder process.")

    total_processed_count = 0

    unprocessed_acts = get_unprocessed_acts()
    if not unprocessed_acts:
        print("No new acts to process. Exiting.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ACTS) as executor:
        futures = {executor.submit(process_act, act_title, i + total_processed_count): (act_title, i + total_processed_count)
                   for i, act_title in enumerate(unprocessed_acts)}

        for future in concurrent.futures.as_completed(futures):
            act_title, current_index = futures[future]
            try:
                result = future.result()
                print(f"Completed processing act {current_index}: {act_title} with result: {result}")
            except Exception as e:
                print(f"A future for act {current_index}: {act_title} raised an exception: {e}")

    total_processed_count += len(unprocessed_acts)

    end_time = datetime.now()
    print(f"\n--- Process Summary ---")
    print(f"Total acts processed in this run: {total_processed_count}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {end_time - start_time}")
    print("-----------------------\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        main()
    finally:
        db_connection.close_all_connections()
        print("All database connections closed.")