import os
import sys
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Ensure project root on path (mirror style from re runner)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topic.topic_extractor_llm import TopicExtractor
from src.db.db_connection import db_connection


load_dotenv()


# --- CONFIGURATION (env-overridable) ---
ACT_TITLE_ENV = os.getenv("TOPIC_ACT") or os.getenv("CORE_ACT")
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini-2.5-flash"
CHUNK_SIZE = int(os.getenv("TOPIC_CHUNK_SIZE", "10000000"))
MAX_WORKERS = int(os.getenv("TOPIC_MAX_WORKERS", "4"))
OUTPUT_DIR = os.getenv("TOPIC_OUTPUT_DIR", "outputs/topics")


def run_single(
    act_title: str,
    *,
    model_name: str = MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    max_workers: int = MAX_WORKERS,
):
    print("Starting topic extraction (single).")
    extractor = TopicExtractor(model_name=model_name, chunk_size=chunk_size, max_workers=max_workers)
    result = extractor.run_for_act(act_title)

    # Add metadata and persist JSON
    now = datetime.now()
    metadata_timestamp = now.strftime("%Y-%m-%d %H-%M-%S")
    filename_timestamp = now.strftime("%Y%m%d_%H%M%S")

    enriched = {
        "model_name": model_name,
        "chunk_size": chunk_size,
        "max_workers": max_workers,
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

    parser = argparse.ArgumentParser(description="Run topic extraction for a single Act.")
    parser.add_argument("--title", default=ACT_TITLE_ENV, help="Act title (env TOPIC_ACT/CORE_ACT if omitted)")
    parser.add_argument("--model", default=MODEL_NAME, help="LLM model (e.g., gemini-2.5-pro)")
    parser.add_argument("--chunk", type=int, default=CHUNK_SIZE)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    model_name = args.model
    chunk_size = args.chunk
    max_workers = args.workers

    print(f"[TopicRun] model={model_name} chunk={chunk_size} workers={max_workers}")

    try:
        act_title = args.title
        if not act_title or not act_title.strip():
            print("ERROR: Act title is required. Use --title or set TOPIC_ACT / CORE_ACT.")
            return
        run_single(
            act_title.strip(),
            model_name=model_name,
            chunk_size=chunk_size,
            max_workers=max_workers,
        )
    finally:
        db_connection.close_all_connections()
        print("Database connections closed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
