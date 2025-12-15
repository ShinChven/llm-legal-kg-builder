import argparse
from typing import List

from src.re.act_relationship_handler import ActRelationshipHandler
from src.db.db_connection import db_connection
from src.db.create_re_layer_counts_table import ensure_re_layer_counts_table


def get_pending_core_acts(layers: int, limit: int = None) -> List[str]:
    """Return core act titles from legislations that are not fully processed for the given layers.

    An act is considered fully processed when it has one row per layer (1..layers)
    in re_layer_counts.
    """
    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Cannot query pending acts.")
        return []

    try:
        with conn.cursor() as cursor:
            query = f"""
                SELECT l.title
                FROM legislations l
                LEFT JOIN (
                    SELECT core_act, COUNT(*) AS cnt
                    FROM re_layer_counts
                    WHERE layer BETWEEN 1 AND %s
                    GROUP BY core_act
                ) r ON r.core_act = l.title
                WHERE COALESCE(r.cnt, 0) < %s
                ORDER BY l.title
            """
            params = (layers, layers)
            if limit is not None and limit > 0:
                query += " LIMIT %s"
                params = (*params, limit)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [r[0] for r in rows]
    except Exception as e:
        print(f"Error querying pending acts: {e}")
        return []
    finally:
        if conn:
            db_connection.release_connection(conn)


def upsert_layer_counts(core_act: str, layers: int, acts_per_layer: dict, rels_per_layer: dict):
    """Upsert per-layer counts (act_count, relationship_count) for one core_act in a single transaction."""
    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Cannot upsert layer counts.")
        return False

    try:
        with conn.cursor() as cursor:
            for layer in range(1, layers + 1):
                act_count = int(acts_per_layer.get(layer, 0))
                relationship_count = int(rels_per_layer.get(layer, 0))
                cursor.execute(
                    """
                    INSERT INTO re_layer_counts (
                        core_act, layer, act_count, relationship_count
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (core_act, layer)
                    DO UPDATE SET
                        act_count = EXCLUDED.act_count,
                        relationship_count = EXCLUDED.relationship_count,
                        computed_at = NOW()
                    ;
                    """,
                    (
                        core_act,
                        layer,
                        act_count,
                        relationship_count,
                    ),
                )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error upserting counts for '{core_act}': {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            db_connection.release_connection(conn)


def process_all(layers: int, limit: int = None):
    """Process all pending acts and store per-layer counts."""
    ensure_re_layer_counts_table()

    handler = ActRelationshipHandler()
    pending = get_pending_core_acts(layers, limit=limit)
    if not pending:
        print("No pending acts to process.")
        return

    print(f"Found {len(pending)} pending act(s) to process for {layers} layer(s).")

    for idx, title in enumerate(pending, start=1):
        print(f"[{idx}/{len(pending)}] Analyzing '{title}'...")
        tree = handler.get_relationship_layers(title, layers)
        if not tree or not tree.get("children"):
            print(f"  No relationships found for '{title}'. Recording zeros.")
            # Upsert zeros so we don't reprocess
            upsert_layer_counts(
                title,
                layers,
                acts_per_layer={i: 0 for i in range(1, layers + 1)},
                rels_per_layer={i: 0 for i in range(1, layers + 1)},
            )
            continue

        stats = handler.get_network_statistics(tree, layers)
        if not stats:
            print(f"  Could not compute statistics for '{title}'. Skipping.")
            continue

        acts_per_layer = stats.get("acts_per_layer", {})
        rels_per_layer = stats.get("relationships_per_layer", {})

        success = upsert_layer_counts(
            title,
            layers,
            acts_per_layer=acts_per_layer,
            rels_per_layer=rels_per_layer,
        )
        if success:
            print(f"  Stored counts for '{title}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="(Deprecated) Compute per-layer relationship/act counts for all Acts.")
    parser.add_argument("layers", nargs="?", type=int, default=5, help="Number of layers to analyze (default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of acts processed in this run.")
    args = parser.parse_args()

    print("WARNING: This script is deprecated. Use src/re/count_relationships_all.py instead, which computes per-layer counts and layer 0 aggregate in one pass.")
    process_all(args.layers, limit=args.limit)
