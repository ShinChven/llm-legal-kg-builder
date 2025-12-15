import argparse
from typing import Set

from src.re.act_relationship_handler import ActRelationshipHandler
from src.db.db_connection import db_connection
from src.db.create_re_layer_counts_table import ensure_re_layer_counts_table


def count_all_relationships(title: str, dry_run: bool = False):
    """
    Traverse the relationship graph starting from `title` until no more new acts
    are found (BFS with cycle protection). Count all relationship codes across
    all reachable edges and upsert the result into `re_layer_counts` with layer=0.

    Notes:
    - Each row in `act_relationships` may contain multiple relationship codes in
      the JSONB `relationships` array; we count each code. If missing/empty, we
      count it as a single generic edge.
    - `act_count` stored for layer=0 is the number of unique reachable acts
      excluding the core act.
    """
    handler = ActRelationshipHandler()

    print(f"Analyzing full relationship tree for '{title}' (no layer limit)...")

    visited: Set[str] = set([title])
    queue = [title]

    total_relationship_codes = 0
    discovered_acts: Set[str] = set()  # excludes core

    while queue:
        current = queue.pop(0)
        rows = handler.get_relationships_by_subject(current) or []

        for row in rows:
            # row: (id, subject_name, object_name, relationships, ...)
            object_name = row[2]
            rel_codes = row[3] or []

            # Count relationship codes for this edge
            if isinstance(rel_codes, list) and len(rel_codes) > 0:
                total_relationship_codes += len(rel_codes)
            else:
                # If relationships is empty/None, treat as a single generic edge
                total_relationship_codes += 1

            # Track discovered acts and continue BFS if not visited
            if object_name not in visited:
                visited.add(object_name)
                discovered_acts.add(object_name)
                queue.append(object_name)

    print("\n--- Aggregated Relationship Statistics ---")
    print(f"Core Act: {title}")
    print(f"Total Unique Acts (excluding core): {len(discovered_acts)}")
    print(f"Total Relationship Codes (all layers): {total_relationship_codes}")

    if dry_run:
        print("\nDry-run: skipping database writes to re_layer_counts.")
        return

    # Persist to re_layer_counts with layer=0
    ensure_re_layer_counts_table()
    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Skipping write to re_layer_counts.")
        return

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO re_layer_counts (core_act, layer, act_count, relationship_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (core_act, layer) DO UPDATE SET
                    act_count = EXCLUDED.act_count,
                    relationship_count = EXCLUDED.relationship_count,
                    computed_at = NOW();
                """,
                (title, 0, int(len(discovered_acts)), int(total_relationship_codes)),
            )
        conn.commit()
        print("\nSaved aggregated counts to re_layer_counts with layer=0.")
    except Exception as e:
        print(f"Error writing aggregated counts to database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_connection.release_connection(conn)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Count all relationships reachable from a given Act and "
            "store the result in re_layer_counts with layer=0."
        )
    )
    parser.add_argument("title", type=str, help="The title of the Act.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print only; do not write to the database.",
    )

    args = parser.parse_args()
    count_all_relationships(args.title, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

