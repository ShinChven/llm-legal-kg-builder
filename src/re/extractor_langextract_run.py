import os
import sys
import yaml
import logging

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.re.extractor_langextract import LangExtractRelationshipExtractor
from src.re.act_relationship_handler import ActRelationshipHandler
from src.db.db_connection import db_connection

# --- CONFIGURATION ---
# Set the title of the Act you want to process here
ACT_TITLE = "Stamp and Cheque Duties Act 1971"
MAX_WORKERS = 10
MODEL_NAME = "gemini-2.5-flash"

# Set to True to generate langextract_visualization.html in the root directory
VISUALIZE_RESULTS = True

OUTPUT_DIR = "outputs/re/langextract"

def main():
    """Main function to run the langextract relationship extraction."""
    logging.info(f"Starting relationship extraction for: '{ACT_TITLE}'")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        act_handler = ActRelationshipHandler()
        existing_relations = act_handler.get_relationships_by_subject(ACT_TITLE)
        if existing_relations:
            print(f"Found {len(existing_relations)} existing relationships for '{ACT_TITLE}'.")
            while True:
                choice = input("Do you want to remove them and re-extract? (yes/no): ").lower().strip()
                if choice in ["yes", "y"]:
                    act_handler.delete_relationships_by_subject(ACT_TITLE)
                    break
                elif choice in ["no", "n"]:
                    print(f"Skipping extraction for '{ACT_TITLE}'.")
                    return
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
        # Initialize and run the extractor
        # The extractor now handles its own database connection and text retrieval.
        extractor = LangExtractRelationshipExtractor(
            model_id=MODEL_NAME,
            max_workers=MAX_WORKERS,
            use_cache=False # Force re-extraction for this run
        )

        # The method now orchestrates the full process, including storing results in the DB.
        # It no longer returns data directly.
        extractor.extract_and_store_relationships(
            title=ACT_TITLE,
            visualize=VISUALIZE_RESULTS,
            output_dir=OUTPUT_DIR
        )

        logging.info(f"Extraction process for '{ACT_TITLE}' completed. Check logs for details.")

    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        # The extractor uses the global singleton connection pool from db_connection.
        # We can close all connections here at the end of the script.
        db_connection.close_all_connections()
        logging.info("Database connections closed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
