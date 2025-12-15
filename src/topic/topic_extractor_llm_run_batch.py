import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from dotenv import load_dotenv

from src.db.db_connection import db_connection
from src.db.create_act_topics_table import ensure_act_topics_table
from src.topic.topic_extractor_llm import TopicExtractor  # type: ignore


load_dotenv()


def process_all_nz_acts(
    max_workers: int = int(os.getenv("TOPIC_BATCH_WORKERS", "3")),
    limit: int | None = None,
    model_name: str | None = os.getenv("MODEL_NAME"),
    chunk_size: int = int(os.getenv("TOPIC_CHUNK_SIZE", "10000000")),
):
    # Fetch distinct titles from both NZ legislation sources
    ensure_act_topics_table()
    conn = db_connection.get_connection()
    if conn is None:
        print("Could not obtain a database connection.")
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT l.title
                FROM legislations AS l
                WHERE l.source IN ('legislation.govt.nz', 'www.legislation.govt.nz')
                  AND l.text IS NOT NULL
                  AND l.title IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM act_topics AS at
                      WHERE at.act_title = l.title
                  )
                ORDER BY l.title
                """
            )
            rows = cursor.fetchall()
            all_titles: List[str] = [r[0] for r in rows]
    except Exception as e:
        print(f"Error fetching act titles: {e}")
        return
    finally:
        db_connection.release_connection(conn)
    if limit is not None:
        all_titles = all_titles[: int(limit)]

    total = len(all_titles)
    if total == 0:
        print("No acts found from sources 'legislation.govt.nz' or 'www.legislation.govt.nz'.")
        return

    print(f"[TopicBatch] Found {total} act(s) to process from NZ legislation sources.")
    if not model_name:
        raise ValueError("No model specified. Set MODEL_NAME or pass --model.")

    def run_one(title: str):
        try:
            extractor = TopicExtractor(model_name=model_name, chunk_size=chunk_size, max_workers=2)
            extractor.run_for_act(title)
            return True
        except Exception as e:
            print(f"[TopicBatch] Error processing '{title}': {e}")
            traceback.print_exc()
            return False

    start = time.time()
    success = 0
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(run_one, t): t for t in all_titles}
        for i, fut in enumerate(as_completed(futures), start=1):
            title = futures[fut]
            ok = fut.result()
            success += 1 if ok else 0
            print(f"[TopicBatch] Progress {i}/{total}: '{title}' done={'yes' if ok else 'no'}")

    elapsed = time.time() - start
    print(f"[TopicBatch] Completed. Success={success}/{total}. Elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process topic extraction for NZ legislation source.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of acts to process")
    parser.add_argument("--workers", type=int, default=int(os.getenv("TOPIC_BATCH_WORKERS", "3")))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--chunk", type=int, default=int(os.getenv("TOPIC_CHUNK_SIZE", "1200")))
    args = parser.parse_args()

    process_all_nz_acts(max_workers=args.workers, limit=args.limit, model_name=args.model, chunk_size=args.chunk)
