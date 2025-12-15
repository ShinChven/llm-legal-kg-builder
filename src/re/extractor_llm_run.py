import os
import sys
import logging
import yaml
from datetime import datetime
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.re.extractor_llm import RelationshipExtractor
from src.re.act_relationship_handler import ActRelationshipHandler
from src.db.db_connection import db_connection

load_dotenv()

# --- CONFIGURATION ---
USE_CACHE = False  # Set to False to skip database cache and always run LLM
CORE_ACT_TITLE = os.getenv("CORE_ACT")
EXTRACTION_LAYERS = 1
MODEL_NAME = 'gemini-2.5-flash'
CHUNK_SIZE = 1000000
MAX_WORKERS = 10
OUTPUT_DIR = "outputs/re/llm"
ENABLE_STRING_MATCHER = True  # Toggle to skip filtering target Acts absent from source text

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
            else:
                convert_relationship_lists(value)
    elif isinstance(node, list):
        for item in node:
            convert_relationship_lists(item)

def setup_yaml_style():
    """Sets up a custom representer for RelationshipList to use flow style."""
    def relationship_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    yaml.add_representer(RelationshipList, relationship_list_representer, Dumper=yaml.SafeDumper)

def main():
    print("Starting recursive relationship extraction process.")

    if not CORE_ACT_TITLE or not CORE_ACT_TITLE.strip():
        print("ERROR: CORE_ACT environment variable is not set. Please define CORE_ACT in your .env file.")
        return

    act_handler = ActRelationshipHandler()

    try:
        extractor = RelationshipExtractor(
            model_name=MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            max_workers=MAX_WORKERS,
            act_handler=act_handler,
            use_cache=USE_CACHE,
            enable_string_matching=ENABLE_STRING_MATCHER,
        )

        extractor.run_recursive(CORE_ACT_TITLE, EXTRACTION_LAYERS)

        relationship_tree = extractor.build_relationship_tree(CORE_ACT_TITLE, EXTRACTION_LAYERS)
        statistics = extractor.get_statistics()

        if relationship_tree and CORE_ACT_TITLE in relationship_tree:
            node = relationship_tree[CORE_ACT_TITLE]
            existing_children = node.pop("children", None)
            node["statistics"] = statistics
            if existing_children:
                node["children"] = existing_children

        # Timestamps
        now = datetime.now()
        filename_timestamp = now.strftime("%Y%m%d_%H%M%S")
        metadata_timestamp = now.strftime("%Y-%m-%d %H-%M-%S")

        output_root = {
            "model_name": MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "max_workers": MAX_WORKERS,
            "extraction_layers": EXTRACTION_LAYERS,
            "core_act_title": CORE_ACT_TITLE,
            "generated_at": metadata_timestamp,
        }
        output_root.update(relationship_tree)

        convert_relationship_lists(output_root)
        setup_yaml_style()

        yaml_output = yaml.dump(
            output_root,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
            Dumper=yaml.SafeDumper
        )

        print("\n--- Relationship Tree (YAML) ---")
        print(yaml_output)
        print("---------------------------------")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_title = CORE_ACT_TITLE.replace(" ", "_").replace("/", "_")
        filename = f"{safe_title}_tree_{filename_timestamp}.yaml"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(yaml_output)
        print(f"Successfully saved relationship tree to {filepath}")

    except Exception as e:  # noqa: BLE001
        print(f"A critical error occurred: {e}")
    finally:
        db_connection.close_all_connections()
        print("Database connections closed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
