import argparse
from typing import Optional, Tuple

from src.db.db_connection import db_connection


def _punct_symbol_count(s: str) -> int:
    """Count non-alphanumeric, non-space characters in a string."""
    return sum(1 for ch in s if not ch.isalnum() and not ch.isspace())


def _choose_canonical(a: str, b: str) -> str:
    """
    Choose the string with fewer punctuations/symbols.
    Tie-breakers: shorter length, then lexicographical order.
    """
    pa, pb = _punct_symbol_count(a), _punct_symbol_count(b)
    if pa != pb:
        return a if pa < pb else b
    la, lb = len(a.strip()), len(b.strip())
    if la != lb:
        return a if la < lb else b
    return a if a <= b else b


def _best_internal_match(cur, row_id: int) -> Optional[Tuple[int, str, float]]:
    """
    Return (match_id, match_title, similarity) for the best internal match within lost_and_found
    for the given row id, based on cosine similarity using pgvector.
    """
    cur.execute(
        """
        SELECT b.id,
               b.object_name,
               1 - (a.title_embedding::vector <=> b.title_embedding::vector) AS similarity
        FROM lost_and_found a
        JOIN lost_and_found b ON b.id <> a.id
        WHERE a.id = %s
          AND a.title_embedding IS NOT NULL
          AND b.title_embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT 1
        """,
        (row_id,),
    )
    m = cur.fetchone()
    if not m:
        return None
    return m[0], m[1], float(m[2])


def internal_compare_update(threshold: float = 0.99, limit: Optional[int] = None) -> None:
    """
    For lost_and_found rows with (similarity IS NULL OR similarity < threshold),
    find best internal match by embedding similarity within the same table. If >= threshold,
    set found_title to the canonical title among the pair (fewer punctuation/symbols) and update similarity.
    Prints matched results as it proceeds.
    """
    conn = db_connection.get_connection()
    if not conn:
        print("Could not obtain a database connection.")
        return

    processed = 0
    updated = 0
    try:
        with conn.cursor() as cur:
            query = (
                """
                SELECT id, object_name, COALESCE(similarity, -1.0) AS sim
                FROM lost_and_found
                WHERE title_embedding IS NOT NULL
                  AND (similarity IS NULL OR similarity < %s)
                ORDER BY id ASC
                """
            )
            params = [threshold]
            if limit is not None and limit > 0:
                query += " LIMIT %s"
                params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()

            if not rows:
                print("No rows eligible for internal comparison.")
                return

            print(f"Scanning {len(rows)} row(s) with similarity < {threshold} for internal matches...")

            for row_id, obj_name, sim in rows:
                processed += 1
                best = _best_internal_match(cur, row_id)
                if not best:
                    print(f"[{processed}] {obj_name}: no internal candidate with embeddings.")
                    continue

                match_id, match_title, match_sim = best
                if match_sim >= threshold:
                    canonical = _choose_canonical(obj_name, match_title)
                    cur.execute(
                        """
                        UPDATE lost_and_found
                        SET found_title = %s,
                            similarity = %s
                        WHERE id = %s
                        """,
                        (canonical, match_sim, row_id),
                    )
                    updated += cur.rowcount
                    print(
                        f"[{processed}] MATCH: '{obj_name}'  <->  '{match_title}'  "
                        f"sim={match_sim:.4f}  chosen='{canonical}'"
                    )
                else:
                    print(
                        f"[{processed}] NO MATCH: '{obj_name}' best='{match_title}' sim={match_sim:.4f} (< {threshold})"
                    )

        conn.commit()
        print(f"Done. Updated {updated} row(s) out of {processed} processed.")

    except Exception as e:
        print(f"An error occurred during internal comparison: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_connection.release_connection(conn)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Internal compare lost_and_found rows by embeddings; when best match >= threshold, "
            "update found_title to the canonical title (fewest punctuation/symbols) and similarity."
        )
    )
    parser.add_argument("--threshold", type=float, default=0.99, help="Similarity threshold (default: 0.99)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process")
    args = parser.parse_args()

    internal_compare_update(threshold=args.threshold, limit=args.limit)


if __name__ == "__main__":
    main()

