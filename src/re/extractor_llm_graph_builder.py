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
    Fetches a list of legislations that have not been processed yet
    and do not have existing relationships. This is done in two steps.
    """
    print(f"Fetching all unprocessed acts from the database...")
    conn = db_connection.get_connection()
    if not conn:
        print("Could not get a database connection.")
        return []

    try:
        with conn.cursor() as cursor:
            # Step 1: Get acts that haven't been processed at all yet.
            # We fetch all acts that are null, and then process them one by one
            # to check if they have relationships.
            cursor.execute("""
                SELECT l.id, l.title
                FROM legislations l
                WHERE l.processed_at IS NULL AND l.word_count > %s AND l.word_count < %s
                ORDER BY l.id ASC;
            """, (MIN_WORD_COUNT, MAX_WORD_COUNT))
            candidate_acts_with_id = cursor.fetchall()

            if not candidate_acts_with_id:
                print("No acts found with processed_at IS NULL.")
                return []

            print(f"Found {len(candidate_acts_with_id)} candidate acts. Now checking for existing relationships and updating processed_at.")

            final_acts_to_process = []
            acts_to_mark_processed = []

            for act_id, act_title in candidate_acts_with_id:
                # Check if the act already has relationships
                cursor.execute("""
                    SELECT 1
                    FROM act_relationships
                    WHERE subject_name = %s
                    LIMIT 1;
                """, (act_title,))

                if cursor.fetchone():
                    # Act has relationships, mark it as processed
                    acts_to_mark_processed.append(act_id)
                else:
                    # Act does not have relationships, add to list for processing
                    final_acts_to_process.append(act_title)

            if acts_to_mark_processed:
                # Update processed_at for acts that already have relationships
                now = datetime.now()
                placeholders = ','.join(['%s'] * len(acts_to_mark_processed))
                cursor.execute(f"""
                    UPDATE legislations
                    SET processed_at = %s
                    WHERE id IN ({placeholders});
                """, (now, *acts_to_mark_processed))
                conn.commit()
                print(f"Marked {len(acts_to_mark_processed)} acts as processed (already had relationships).")

            print(f"Found {len(final_acts_to_process)} unprocessed acts to be processed.")
            return final_acts_to_process
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
