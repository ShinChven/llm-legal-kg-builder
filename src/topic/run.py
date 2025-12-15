import os
import sys
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Ensure project root on path (mirror style from re runner)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topic.topic_extractor_llm import TopicExtractor
from src.topic.topic_extractor_llm_run_batch import process_all_nz_acts

from src.db.db_connection import db_connection


load_dotenv()


# --- CONFIGURATION (env-overridable) ---
MODE = os.getenv("TOPIC_MODE", "single").strip().lower()  # 'single' | 'batch'
ACT_TITLE_ENV = os.getenv("TOPIC_ACT") or os.getenv("CORE_ACT")
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini-2.5-flash"
CHUNK_SIZE = int(os.getenv("TOPIC_CHUNK_SIZE", "1200"))
MAX_WORKERS = int(os.getenv("TOPIC_MAX_WORKERS", "4"))
OUTPUT_DIR = os.getenv("TOPIC_OUTPUT_DIR", "outputs/topics")


def run_single(act_title: str):
    print("Starting topic extraction (single mode).")
    extractor = TopicExtractor(model_name=MODEL_NAME, chunk_size=CHUNK_SIZE, max_workers=MAX_WORKERS)
    result = extractor.run_for_act(act_title)

    # Add metadata and persist JSON
    now = datetime.now()
    metadata_timestamp = now.strftime("%Y-%m-%d %H-%M-%S")
    filename_timestamp = now.strftime("%Y%m%d_%H%M%S")

    enriched = {
        "model_name": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "max_workers": MAX_WORKERS,
        "generated_at": metadata_timestamp,
        **result,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_title = act_title.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{safe_title}_topics_{filename_timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"Saved topics JSON to {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Topic Extraction for a single Act or batch.")
    parser.add_argument("--mode", choices=["single", "batch"], default=MODE)
    parser.add_argument("--title", default=ACT_TITLE_ENV, help="Act title for single mode (env TOPIC_ACT/CORE_ACT if omitted)")
    parser.add_argument("--model", default=MODEL_NAME, help="LLM model (e.g., gemini-2.5-pro)")
    parser.add_argument("--chunk", type=int, default=CHUNK_SIZE)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for batch mode")
    args = parser.parse_args()

    global MODEL_NAME, CHUNK_SIZE, MAX_WORKERS
    MODEL_NAME = args.model
    CHUNK_SIZE = args.chunk
    MAX_WORKERS = args.workers

    print(
        f"[TopicRun] mode={args.mode} model={MODEL_NAME} chunk={CHUNK_SIZE} workers={MAX_WORKERS}"
    )

    try:
        if args.mode == "single":
            act_title = args.title
            if not act_title or not act_title.strip():
                print("ERROR: Act title is required in single mode. Use --title or set TOPIC_ACT / CORE_ACT.")
                return
            run_single(act_title.strip())
        else:
            process_all_nz_acts(max_workers=MAX_WORKERS, limit=args.limit, model_name=MODEL_NAME, chunk_size=CHUNK_SIZE)
    finally:
        db_connection.close_all_connections()
        print("Database connections closed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()

