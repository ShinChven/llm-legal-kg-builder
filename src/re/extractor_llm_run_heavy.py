import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Set
import yaml
from dotenv import load_dotenv

from src.re.extractor_llm import RelationshipExtractor
from src.re.reassurance_extractor import ReassuranceExtractor
from src.re.act_relationship_handler import ActRelationshipHandler
from src.db.db_connection import db_connection
from src.re.normalization import normalize_title

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()


# --- CONFIGURATION ---
USE_CACHE = False  # Set to False to skip database cache and always run LLM
CLEAN_RUN = True  # If True, remove all existing act_relationships with subject_title = core_act_title before processing
CORE_ACT_TITLE = os.getenv("CORE_ACT")
MODEL_NAME = 'gemini-2.5-flash'
# Random non-overlapping chunking strategy to increase the possibility of finding more relationships
CHUNK_SIZES = [
    250,
    720,
    # 500, 600, 1000
    1150,
	# 5300,
    1000000,
]  # Array of chunk sizes for multiple runs
MAX_RELATIONSHIP_WORKERS = 10  # strongly suggest to use a low number for long text extractions
MAX_REASSURANCE_WORKERS = 1  # adjust independently for reassurance phase if needed
REASSURANCE_BATCH_SIZE = 5  # Number of acts to reassure in a single LLM call
OUTPUT_DIR = "outputs/re/llm"
ENABLE_STRING_MATCHER = False  # Toggle to skip filtering target Acts absent from source text

# --- Custom YAML Formatting ---


class RelationshipList(list):
    """A custom list type to signal flow-style YAML dumping."""
    pass


def convert_relationship_lists(node):
    """Recursively converts lists under 'relationships' key to RelationshipList."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == 'relationships' and isinstance(value, list):
                node[key] = RelationshipList(value)
            elif key == 'children' and isinstance(value, dict):
                # Handle the children structure
                for child_key, child_value in value.items():
                    convert_relationship_lists(child_value)
            else:
                convert_relationship_lists(value)
    elif isinstance(node, list):
        for item in node:
            convert_relationship_lists(item)


def setup_yaml_style():
    """Sets up a custom representer for RelationshipList to use flow style."""
    def relationship_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    yaml.add_representer(
        RelationshipList, relationship_list_representer, Dumper=yaml.SafeDumper)


class NoAnchorSafeDumper(yaml.SafeDumper):
    """A SafeDumper that disables anchor/alias generation to avoid &id001 references."""

    def ignore_aliases(self, data):
        return True


def check_existing_relationships_and_prompt(act_handler: ActRelationshipHandler, core_act_title: str) -> bool:
    """
    Check if relationships exist for the core act and prompt user for deletion confirmation.

    Returns:
        bool: True if deletion should proceed, False otherwise
    """
    # Safety check: ensure core_act_title is not empty or None
    if not core_act_title or not core_act_title.strip():
        print("ERROR: core_act_title is empty or None. Aborting to prevent accidental deletion of all records.")
        return False

    # Check if any existing relationships exist for this subject
    conn = act_handler.get_connection()
    if conn is None:
        print("WARNING: Cannot connect to database to check existing relationships.")
        return False

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM act_relationships WHERE subject_name = %s",
                (core_act_title,)
            )
            existing_count = cursor.fetchone()[0]

            if existing_count == 0:
                print(f"No existing relationships found for subject '{core_act_title}'. No deletion needed.")
                return False

            print(f"Found {existing_count} existing relationships for subject '{core_act_title}'.")

            # Prompt for confirmation
            while True:
                response = input(f"Do you want to delete these {existing_count} existing relationships? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    print("User confirmed deletion.")
                    return True
                elif response in ['n', 'no', '']:
                    print("User declined deletion. Existing relationships will be overwritten during merge.")
                    return False
                else:
                    print("Please answer 'y' for yes or 'n' for no.")

    except Exception as e:
        print(f"Error checking existing relationships: {e}")
        return False
    finally:
        act_handler.release_connection(conn)


def merge_and_deduplicate_relationships(all_results: List[Dict]) -> Dict:
    """
    Merges and deduplicates relationships from multiple extraction runs.

    Args:
        all_results: List of dictionaries containing target_documents from each run

    Returns:
        Dictionary with merged and deduplicated relationships
    """
    merged_relationships = {}

    for result in all_results:
        target_documents = result.get('target_documents', {})

        for act_title, relationships in target_documents.items():
            if act_title not in merged_relationships:
                merged_relationships[act_title] = set()

            # Add relationships to set for deduplication
            if isinstance(relationships, list):
                merged_relationships[act_title].update(relationships)
            else:
                # Handle case where relationships might be a single value
                merged_relationships[act_title].add(relationships)

    # Convert sets back to lists
    for act_title in merged_relationships:
        merged_relationships[act_title] = list(merged_relationships[act_title])

    return merged_relationships


def main():
    print("Starting heavy recursive relationship extraction process with multiple chunk sizes.")
    print(f"Chunk sizes to process: {CHUNK_SIZES}")

    act_handler = ActRelationshipHandler()

    # Safety check: ensure CORE_ACT_TITLE is not empty
    if not CORE_ACT_TITLE or not CORE_ACT_TITLE.strip():
        print("ERROR: CORE_ACT_TITLE is empty or None. Please set a valid core act title.")
        return

    # Check existing relationships and prompt for deletion if CLEAN_RUN is enabled
    should_delete_existing = False
    if CLEAN_RUN:
        print(f"CLEAN_RUN enabled: Checking existing relationships for subject '{CORE_ACT_TITLE}'")
        should_delete_existing = check_existing_relationships_and_prompt(act_handler, CORE_ACT_TITLE)

        if should_delete_existing:
            act_handler.delete_relationships_by_subject(CORE_ACT_TITLE)
    else:
        print("CLEAN_RUN disabled: Keeping existing relationships (they will be overwritten during merge)")

    all_extraction_results = []
    run_statistics = []

    try:
        # If CHUNK_SIZES is non-empty, run extraction for each chunk size and merge results.
        if CHUNK_SIZES:
            # Run extraction for each chunk size
            for i, chunk_size in enumerate(CHUNK_SIZES):
                print(
                    f"\n=== Run {i+1}/{len(CHUNK_SIZES)}: Processing with chunk_size={chunk_size} ===")

                extractor = RelationshipExtractor(
                    model_name=MODEL_NAME,
                    chunk_size=chunk_size,
                    max_workers=MAX_RELATIONSHIP_WORKERS,
                    act_handler=act_handler,
                    use_cache=USE_CACHE,
                    enable_string_matching=ENABLE_STRING_MATCHER,
                )

                # For heavy runner, we only extract for the core act without recursion
                # This avoids complications with multiple layers and focuses on accuracy
                print(f"Extracting relationships for core act: '{CORE_ACT_TITLE}'")

                try:
                    # Extract for single act
                    extraction_result = extractor._extract_for_single_act(
                        title=CORE_ACT_TITLE,
                        enable_logging=True,
                        act_handler=act_handler
                    )

                    all_extraction_results.append(extraction_result)

                    # Collect statistics for this run
                    target_documents = extraction_result.get(
                        'target_documents', {})
                    total_acts_found = len(target_documents)
                    total_relationships = sum(len(rels)
                                              for rels in target_documents.values())

                    run_stats = {
                        'chunk_size': chunk_size,
                        'acts_found': total_acts_found,
                        'total_relationships': total_relationships,
                        'target_acts': list(target_documents.keys()) if target_documents else []
                    }
                    run_statistics.append(run_stats)

                    print(
                        f"Run {i+1} completed: Found {total_acts_found} target acts with {total_relationships} total relationships")

                except Exception as e:
                    print(f"Error in run {i+1} with chunk_size={chunk_size}: {e}")
                    run_statistics.append({
                        'chunk_size': chunk_size,
                        'error': str(e),
                        'acts_found': 0,
                        'total_relationships': 0,
                        'target_acts': []
                    })

            # Merge and deduplicate results from all runs
            print(
                f"\n=== Merging and deduplicating results from {len(CHUNK_SIZES)} runs ===")
            merged_relationships = merge_and_deduplicate_relationships(
                all_extraction_results)

            print(
                f"Merged results: Found {len(merged_relationships)} unique target acts")
            for act_title, relationships in merged_relationships.items():
                print(f"  - {act_title}: {len(relationships)} relationships")

            # --- Store initial merged results in database ---
            if merged_relationships:
                print("\n=== Storing initial merged results in database before reassurance... ===")
                # This loop creates the records that the reassurance step will update.
                for object_name, relationships in merged_relationships.items():
                    normalized_object_name = normalize_title(object_name)
                    act_handler.upsert_relationship(
                        CORE_ACT_TITLE, normalized_object_name, relationships)
                print("Initial results stored.")
            else:
                print("\nNo relationships found to store or reassure.")

        else:
            # CHUNK_SIZES is empty: skip extraction runs and proceed straight to DB query + reassurance
            print("\nCHUNK_SIZES is empty: skipping extraction/merge/store steps and proceeding to DB-based reassurance.")
            merged_relationships = {}

        # --- Reassurance Step ---
        print("\n=== Starting Reassurance Process ===")

        # Query the database to get the aggregated data to be reassured.
        print("\nQuerying database for aggregated data to begin reassurance...")
        relationships_from_db = act_handler.get_relationships_by_subject(
            CORE_ACT_TITLE)

        # Convert db result to the dictionary format needed for reassurance
        relationships_to_reassure = {
            rel[2]: rel[3] for rel in relationships_from_db
        }
        print(
            f"Successfully fetched {len(relationships_to_reassure)} relationships from DB to start reassurance.")

        if relationships_to_reassure:
            reassurer = ReassuranceExtractor(
                model_name=MODEL_NAME,
                reassurance_batch_size=REASSURANCE_BATCH_SIZE,
                max_workers=MAX_REASSURANCE_WORKERS,
                act_handler=act_handler
            )
            # The reassure_relationships method now also updates the DB
            reassured_relationships = reassurer.reassure_relationships(
                source_act_title=CORE_ACT_TITLE,
                target_relationships=relationships_to_reassure,  # Use data from DB
                enable_logging=True
            )
            print(
                f"Reassurance complete. Confirmed and updated {len(reassured_relationships)} relationships in the database.")
        else:
            print("No relationships found in the database to reassure.")

        # --- Final Reporting Step ---
        # Re-query the database to get the final, reassured data for the report.
        print("\nRe-querying database for reassured data to build final report...")
        final_relationships_from_db = act_handler.get_relationships_by_subject(CORE_ACT_TITLE)

        # Convert db result to the dictionary format needed for the YAML output
        final_report_data = {
            rel[2]: rel[3] for rel in final_relationships_from_db
        }
        print(f"Successfully fetched {len(final_report_data)} reassured relationships from DB for reporting.")


        # Build final output structure to match original format
        children_dict = {}
        for act_title, relationships in final_report_data.items():
            children_dict[act_title] = {
                "relationships": relationships
            }

        relationship_tree = {
            CORE_ACT_TITLE: {
                "children": children_dict,
                "statistics": {
                    "total_target_acts": len(final_report_data),
                    "total_relationships": sum(len(rels) for rels in final_report_data.values()),
                    "run_statistics": run_statistics,
                    "chunk_sizes_used": CHUNK_SIZES,
                    "runs_completed": len([stats for stats in run_statistics if 'error' not in stats])
                }
            }
        }

        # Timestamps
        now = datetime.now()
        filename_timestamp = now.strftime("%Y%m%d_%H%M%S")
        metadata_timestamp = now.strftime("%Y-%m-%d %H-%M-%S")

        output_root = {
            "model_name": MODEL_NAME,
            "chunk_sizes": CHUNK_SIZES,
            "max_relationship_workers": MAX_RELATIONSHIP_WORKERS,
            "max_reassurance_workers": MAX_REASSURANCE_WORKERS,
            "core_act_title": CORE_ACT_TITLE,
            "generated_at": metadata_timestamp,
            "extraction_type": "heavy"
        }
        output_root.update(relationship_tree)

        convert_relationship_lists(output_root)
        setup_yaml_style()

        # Register the relationship list representer with our custom dumper
        def relationship_list_representer(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        yaml.add_representer(
            RelationshipList, relationship_list_representer, Dumper=NoAnchorSafeDumper)

        yaml_output = yaml.dump(
            output_root,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
            Dumper=NoAnchorSafeDumper
        )

        print("\n--- Heavy Extraction Results (YAML) ---")
        print(yaml_output)
        print("---------------------------------")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_title = CORE_ACT_TITLE.replace(" ", "_").replace("/", "_")
        filename = f"{safe_title}_heavy_tree_{filename_timestamp}.yaml"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(yaml_output)
        print(f"Successfully saved heavy extraction results to {filepath}")

    except Exception as e:  # noqa: BLE001
        print(f"A critical error occurred: {e}")
    finally:
        db_connection.close_all_connections()
        print("Database connections closed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main()
