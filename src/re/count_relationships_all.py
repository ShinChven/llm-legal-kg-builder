import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Iterable

from src.db.db_connection import db_connection
from src.db.create_re_layer_counts_table import ensure_re_layer_counts_table
from src.re.count_relationships import count_all_relationships


def get_core_acts_pending_agg(limit: int = None, reprocess: bool = False) -> List[str]:
    """Return act titles that need aggregated counts (layer=0).

    - When reprocess is False: return acts missing a row in re_layer_counts for layer=0.
    - When reprocess is True: return all acts from legislations.
    """
    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Cannot query acts.")
        return []

    try:
        with conn.cursor() as cur:
            if reprocess:
                query = "SELECT title FROM legislations ORDER BY title"
                params: Tuple = tuple()
            else:
                query = (
                    """
                    SELECT l.title
                    FROM legislations l
                    LEFT JOIN re_layer_counts r
                      ON r.core_act = l.title AND r.layer = 0
                    WHERE r.core_act IS NULL
                    ORDER BY l.title
                    """
                )
                params = tuple()
            if limit is not None and limit > 0:
                query += " LIMIT %s"
                params = (*params, limit)
            cur.execute(query, params)
            rows = cur.fetchall()
            return [r[0] for r in rows]
    except Exception as e:
        print(f"Error querying core acts: {e}")
        return []
    finally:
        db_connection.release_connection(conn)


def load_relationship_adjacency() -> Dict[str, List[Tuple[str, Iterable[str]]]]:
    """Load the entire act_relationships table into memory as an adjacency list.

    Returns a dict: subject -> list of (object, relationship_codes_list)
    """
    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Cannot load relationships.")
        return {}

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT subject_name, object_name, relationships
                FROM act_relationships
                """
            )
            adj: Dict[str, List[Tuple[str, Iterable[str]]]] = {}
            for subject, obj, rels in cur.fetchall():
                lst = adj.setdefault(subject, [])
                lst.append((obj, rels or []))
            return adj
    except Exception as e:
        print(f"Error loading relationships into memory: {e}")
        return {}
    finally:
        db_connection.release_connection(conn)


def compute_layered_counts_from_adjacency(
    core: str, adj: Dict[str, List[Tuple[str, Iterable[str]]]]
) -> Tuple[Dict[int, int], Dict[int, int], int, int]:
    """Compute per-layer and total counts using the in-memory adjacency.

    Returns a tuple:
      - acts_per_layer: dict[layer -> unique act count first discovered at this layer]
      - rels_per_layer: dict[layer -> relationship code count of edges from previous layer]
      - total_unique_acts_excl_core: int
      - total_relationship_codes: int

    Notes:
      - Each row's relationships array contributes len(array) codes; if empty/None, counts as 1.
      - Layers start at 1 (edges from layer 0=root to layer 1 nodes).
    """
    visited = {core}
    frontier = [core]

    acts_per_layer: Dict[int, int] = {}
    rels_per_layer: Dict[int, int] = {}
    total_relationship_codes = 0
    total_unique_acts_excl_core = 0

    layer = 0
    while frontier:
        next_frontier: List[str] = []
        layer += 1

        # Initialize counters for this layer
        acts_per_layer.setdefault(layer, 0)
        rels_per_layer.setdefault(layer, 0)

        for u in frontier:
            for v, rels in adj.get(u, []):
                # Relationship code counting for edges going out of this layer
                if rels and isinstance(rels, list):
                    rels_per_layer[layer] += len(rels)
                    total_relationship_codes += len(rels)
                else:
                    rels_per_layer[layer] += 1
                    total_relationship_codes += 1

                if v not in visited:
                    visited.add(v)
                    acts_per_layer[layer] += 1
                    total_unique_acts_excl_core += 1
                    next_frontier.append(v)

        frontier = next_frontier

    # Remove trailing zero layers (if any) for cleanliness
    acts_per_layer = {k: v for k, v in acts_per_layer.items() if v != 0 or rels_per_layer.get(k, 0) != 0}
    rels_per_layer = {k: v for k, v in rels_per_layer.items() if v != 0 or acts_per_layer.get(k, 0) != 0}

    return acts_per_layer, rels_per_layer, total_unique_acts_excl_core, total_relationship_codes


def _upsert_layer0(core: str, act_count: int, rel_count: int):
    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("PostgreSQL connection not available for upsert.")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO re_layer_counts (core_act, layer, act_count, relationship_count)
                VALUES (%s, 0, %s, %s)
                ON CONFLICT (core_act, layer) DO UPDATE SET
                    act_count = EXCLUDED.act_count,
                    relationship_count = EXCLUDED.relationship_count,
                    computed_at = NOW();
                """,
                (core, int(act_count), int(rel_count)),
            )
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        db_connection.release_connection(conn)


def _upsert_per_layers(core: str, acts_per_layer: Dict[int, int], rels_per_layer: Dict[int, int]):
    """Upsert per-layer rows (layer >= 1) into re_layer_counts."""
    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("PostgreSQL connection not available for upsert.")

    try:
        with conn.cursor() as cur:
            for layer, act_cnt in acts_per_layer.items():
                rel_cnt = int(rels_per_layer.get(layer, 0))
                cur.execute(
                    """
                    INSERT INTO re_layer_counts (core_act, layer, act_count, relationship_count)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (core_act, layer) DO UPDATE SET
                        act_count = EXCLUDED.act_count,
                        relationship_count = EXCLUDED.relationship_count,
                        computed_at = NOW();
                    """,
                    (core, int(layer), int(act_cnt), rel_cnt),
                )
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        db_connection.release_connection(conn)


def _process_one(idx: int, title: str, adj: Dict[str, List[Tuple[str, Iterable[str]]]], dry_run: bool = False) -> Tuple[int, str, bool, str]:
    """Process one act using in-memory adjacency; returns (index, title, success, message)."""
    try:
        acts_per_layer, rels_per_layer, total_acts, total_rels = compute_layered_counts_from_adjacency(title, adj)
        if not dry_run:
            if acts_per_layer or rels_per_layer:
                _upsert_per_layers(title, acts_per_layer, rels_per_layer)
            _upsert_layer0(title, total_acts, total_rels)
        return (idx, title, True, "ok")
    except Exception as e:  # protect the thread pool
        return (idx, title, False, str(e))


def process_all(max_workers: int = 1, limit: int = None, reprocess: bool = False, dry_run: bool = False):
    """Process aggregated relationship counts for all (or pending) acts with threading.

    Writes a single row per act to `re_layer_counts` with `layer=0` unless dry_run.
    """
    if max_workers < 1:
        max_workers = 1

    # Clamp workers to DB pool max to avoid exhaustion
    try:
        _, pool_max = db_connection.get_pool_limits()
        if max_workers > pool_max:
            print(f"Requested {max_workers} workers exceeds DB pool max={pool_max}. Using {pool_max}.")
            max_workers = pool_max
    except Exception:
        # If pool doesn't expose limits, keep the requested value
        pass

    # Ensure table exists once up-front (used by both selects and upserts)
    ensure_re_layer_counts_table()

    # Load all relationships once into memory for fast per-act traversal
    adjacency = load_relationship_adjacency()
    if not adjacency:
        print("Adjacency is empty. Proceeding, but results may all be zeros.")

    titles = get_core_acts_pending_agg(limit=limit, reprocess=reprocess)
    if not titles:
        print("No acts to process.")
        return

    total = len(titles)
    print(f"Planning to process {total} act(s) with max_workers={max_workers}.")

    completed = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit with stable 1-based index so we can report [i/total]
        futures = {executor.submit(_process_one, i, t, adjacency, dry_run): (i, t) for i, t in enumerate(titles, start=1)}
        for fut in as_completed(futures):
            idx, title, ok, msg = fut.result()
            if ok:
                completed += 1
                print(f"[{idx}/{total}] [OK] {title}")
            else:
                failures += 1
                print(f"[{idx}/{total}] [FAIL] {title}: {msg}")

    print(f"\nDone. Completed: {completed}, Failed: {failures}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute per-layer relationship counts and total aggregate (layer=0) "
            "for all or pending acts using a thread pool."
        )
    )
    # Support multiple option strings; default is 1 (serial processing)
    ap.add_argument(
        "-j", "--max-workers", "--max-worker",
        dest="max_workers", type=int, default=1,
        help="Number of worker threads (default: 1)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Limit number of acts to process")
    ap.add_argument(
        "--reprocess",
        action="store_true",
        help="Recompute for all acts (ignore existing layer=0 rows)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print only; do not write to the database.",
    )
    args = ap.parse_args()

    process_all(
        max_workers=args.max_workers,
        limit=args.limit,
        reprocess=args.reprocess,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
