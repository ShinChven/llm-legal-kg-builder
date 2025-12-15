import argparse
from src.re.act_relationship_handler import ActRelationshipHandler
from src.db.db_connection import db_connection
from src.db.create_re_layer_counts_table import ensure_re_layer_counts_table


def count_relationships(title: str, layers: int, dry_run: bool = False):
    """
    Counts the number of acts at each layer of relationships for a given Act.

    :param title: The title of the Act to start with.
    :param layers: The number of relationship layers to analyze.
    """
    handler = ActRelationshipHandler()

    print(f"Analyzing {layers} layer(s) of relationships for '{title}'...")

    # First, get the relationship data as a tree
    relationship_tree = handler.get_relationship_layers(title, layers)

    if not relationship_tree or 'children' not in relationship_tree:
        print("No relationships found or tree is empty.")
        return

    # Then, compute statistics on the tree
    stats = handler.get_network_statistics(relationship_tree, layers)

    if not stats:
        print("Could not compute statistics.")
        return

    print("\n--- Relationship Statistics ---")
    print(f"Core Act: {stats.get('core_act')}")
    print(f"Layers Requested: {stats.get('layers_requested')}")
    print(f"Layers Found: {stats.get('layers_found')}")
    print(f"Total Unique Acts: {stats.get('total_acts')}")
    print(f"Total Relationships: {stats.get('total_relationships')}")

    print("\n--- Acts per Layer ---")
    acts_per_layer = stats.get('acts_per_layer', {})
    if not acts_per_layer:
        print("No acts found in any layer.")
    else:
        for layer, count in sorted(acts_per_layer.items()):
            print(f"Layer {layer}: {count} act(s)")

    print("\n--- Relationships per Layer ---")
    rels_per_layer = stats.get('relationships_per_layer', {})
    if not rels_per_layer:
        print("No relationships found in any layer.")
    else:
        for layer, count in sorted(rels_per_layer.items()):
            print(f"Layer {layer}: {count} relationship code(s)")

    # --- Persist per-layer counts (overwrite existing) ---
    if dry_run:
        print("\nDry-run: skipping database writes to re_layer_counts.")
    else:
        ensure_re_layer_counts_table()
        conn = db_connection.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Skipping write to re_layer_counts.")
            return

        try:
            with conn.cursor() as cursor:
                for layer in range(1, layers + 1):
                    act_count = int(acts_per_layer.get(layer, 0))
                    relationship_count = int(rels_per_layer.get(layer, 0))
                    cursor.execute(
                        """
                        INSERT INTO re_layer_counts (core_act, layer, act_count, relationship_count)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (core_act, layer) DO UPDATE SET
                            act_count = EXCLUDED.act_count,
                            relationship_count = EXCLUDED.relationship_count,
                            computed_at = NOW();
                        """,
                        (title, layer, act_count, relationship_count),
                    )
            conn.commit()
            print("\nSaved per-layer counts to re_layer_counts (overwritten if existed).")
        except Exception as e:
            print(f"Error writing counts to database: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                db_connection.release_connection(conn)

    print("\n--- Relationships per Layer ---")
    rels_per_layer = stats.get('relationships_per_layer', {})
    if not rels_per_layer:
        print("No relationships found in any layer.")
    else:
        for layer, count in sorted(rels_per_layer.items()):
            print(f"Layer {layer}: {count} relationship code(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count the number of acts at each relationship layer for a given Act."
    )
    parser.add_argument("title", type=str, help="The title of the Act.")
    parser.add_argument("layers", type=int, help="The number of layers to analyze.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print only; do not write to the database.")

    args = parser.parse_args()

    count_relationships(args.title, args.layers, dry_run=args.dry_run)
