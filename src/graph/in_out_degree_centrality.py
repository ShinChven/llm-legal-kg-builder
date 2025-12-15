"""
Degree centrality over act relationships (directed, weighted, JSONB-aware).

Overview
--------
Degree centrality measures how many connections a node has. For directed
graphs built from `act_relationships` (subject -> object), we support:

    - in:   sum of incoming edge weights
    - out:  sum of outgoing edge weights
    - both: in + out (treat as undirected strength)

Weights reflect the multiplicity of relationship types per pair, consistent
with `katz_centrality.py`:

    - multiplicity (default): number of relationship types
    - unweighted:             1 per (subject, object) pair
    - log:                    1 + log(1 + multiplicity)

Output
------
Writes CSV to `outputs/re/centrality/` with columns:

    Act title, Degree centrality, Explanation

sorted by descending centrality. The filename includes the direction, weight
scheme, and a timestamp.
"""

import argparse
import csv
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple

from src.db.db_connection import db_connection


def build_weighted_graph(weight_scheme: str):
    """Load edges from DB and build adjacency maps and node set.

    Returns:
        titles: list of node titles (sorted), index corresponds to node id
        out_adj: dict[node_id] -> list[(neighbor_id, weight)]
        in_adj:  dict[node_id] -> list[(neighbor_id, weight)]
    """
    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("PostgreSQL connection not available.")

    subjects: List[str] = []
    objects: List[str] = []
    edges_raw: List[Tuple[str, str, list]] = []

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT subject_name, object_name, relationships
                FROM act_relationships
                ORDER BY subject_name, object_name
                """
            )
            for s, o, rels in cur.fetchall():
                subjects.append(s)
                objects.append(o)
                rels_list = rels or []
                edges_raw.append((s, o, rels_list))
    finally:
        db_connection.release_connection(conn)

    titles = sorted(set(subjects) | set(objects))
    idx_of: Dict[str, int] = {t: i for i, t in enumerate(titles)}

    out_adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(titles))}
    in_adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(titles))}

    # Combine multiplicities per pair if multiple rows exist (defensive)
    # We first accumulate into a dict to avoid duplicate parallel edges.
    agg: Dict[Tuple[int, int], float] = {}
    for s, o, rels_list in edges_raw:
        if s == o:
            continue  # ignore self-loops
        i = idx_of[s]
        j = idx_of[o]
        m = max(1, len(rels_list))
        if weight_scheme == "unweighted":
            w = 1.0
        elif weight_scheme == "log":
            w = 1.0 + math.log(1.0 + m)
        else:
            w = float(m)
        agg[(i, j)] = agg.get((i, j), 0.0) + w

    for (i, j), w in agg.items():
        out_adj[i].append((j, w))
        in_adj[j].append((i, w))

    return titles, out_adj, in_adj


def compute_degree(
    titles: List[str],
    out_adj: Dict[int, List[Tuple[int, float]]],
    in_adj: Dict[int, List[Tuple[int, float]]],
    direction: str,
    normalize: bool,
):
    """Compute degree centrality per node.

    Args:
        titles: node titles
        out_adj, in_adj: adjacency lists with weights
        direction: one of {"in", "out", "both"}
        normalize: if True, divide by (n-1) for comparability
    Returns:
        List of (title, score, explanation)
    """
    n = len(titles)
    denom = max(1, n - 1) if normalize else 1

    def top_k_contrib(node_id: int, mode: str, k: int = 3) -> List[Tuple[str, float, float]]:
        # returns list of (neighbor_title, contribution, weight)
        pairs: List[Tuple[int, float]]
        if mode == "in":
            pairs = in_adj[node_id]
            # neighbor is predecessor
            contribs = [(src, w) for (src, w) in pairs]
        elif mode == "out":
            pairs = out_adj[node_id]
            contribs = [(dst, w) for (dst, w) in pairs]
        else:  # both: combine incoming and outgoing
            accum: Dict[int, float] = {}
            for (src, w) in in_adj[node_id]:
                accum[src] = accum.get(src, 0.0) + w
            for (dst, w) in out_adj[node_id]:
                accum[dst] = accum.get(dst, 0.0) + w
            contribs = list(accum.items())

        contribs.sort(key=lambda x: x[1], reverse=True)
        out: List[Tuple[str, float, float]] = []
        for neigh_id, w in contribs[:k]:
            out.append((titles[neigh_id], w / denom, w)) if normalize else out.append((titles[neigh_id], w, w))
        return out

    rows: List[Tuple[str, float, str]] = []
    for i, t in enumerate(titles):
        if direction == "in":
            score_raw = sum(w for _, w in in_adj[i])
        elif direction == "out":
            score_raw = sum(w for _, w in out_adj[i])
        else:  # both
            score_raw = sum(w for _, w in in_adj[i]) + sum(w for _, w in out_adj[i])
        score = score_raw / denom

        tc = top_k_contrib(i, direction)
        tc_str = ", ".join(
            f"{nbr}({contrib:.3g},w={int(w) if abs(w - int(w)) < 1e-9 else w:.3g})" for nbr, contrib, w in tc
        ) if tc else ""
        parts: List[str] = []
        parts.append(f"mode={direction}")
        parts.append(f"normalized={normalize}")
        if tc_str:
            parts.append(f"top: {tc_str}")
        expl = "; ".join(parts)
        rows.append((t, score, expl))

    rows.sort(key=lambda x: (-x[1], x[0]))
    return rows


def write_csv(path: str, rows: List[Tuple[str, float, str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Act title", "Degree centrality", "Explanation"])
        for t, score, expl in rows:
            w.writerow([t, f"{score:.10f}", expl])


def main():
    ap = argparse.ArgumentParser(description="Compute degree centrality from act_relationships and export CSV.")
    ap.add_argument("--weight", choices=["multiplicity", "unweighted", "log"], default="multiplicity", help="Edge weight scheme.")
    ap.add_argument("--direction", choices=["in", "out", "both"], default="in", help="Degree mode: in/out/both. 'both' outputs a combined CSV with in/out/both columns.")
    ap.add_argument("--normalize", action="store_true", help="Normalize by (n-1) similar to NetworkX degree_centrality.")
    args = ap.parse_args()

    titles, out_adj, in_adj = build_weighted_graph(args.weight)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", "re", "centrality")

    if args.direction == "both":
        # Compute in, out, and both degrees and write a combined CSV
        n = len(titles)
        denom = max(1, n - 1) if args.normalize else 1
        in_scores = [sum(w for _, w in in_adj[i]) / denom for i in range(n)]
        out_scores = [sum(w for _, w in out_adj[i]) / denom for i in range(n)]
        both_scores = [in_scores[i] + out_scores[i] for i in range(n)]

        # Sort by both-degree desc, then title
        order = sorted(range(n), key=lambda i: (-both_scores[i], titles[i]))

        out_path = os.path.join(
            out_dir,
            f"degree_centrality_both_{args.weight}_{'norm' if args.normalize else 'raw'}_{ts}.csv",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Act title", "In-degree", "Out-degree", "Both-degree"])  # combined view
            for i in order:
                w.writerow([
                    titles[i],
                    f"{in_scores[i]:.10f}",
                    f"{out_scores[i]:.10f}",
                    f"{both_scores[i]:.10f}",
                ])

        print(f"Saved Degree centrality to: {out_path}")
        print(f"direction=both(combined), weight={args.weight}, normalize={args.normalize}")
    else:
        rows = compute_degree(titles, out_adj, in_adj, args.direction, args.normalize)
        out_path = os.path.join(
            out_dir,
            f"degree_centrality_{args.direction}_{args.weight}_{'norm' if args.normalize else 'raw'}_{ts}.csv",
        )
        write_csv(out_path, rows)

        print(f"Saved Degree centrality to: {out_path}")
        print(f"direction={args.direction}, weight={args.weight}, normalize={args.normalize}")


if __name__ == "__main__":
    try:
        main()
    finally:
        db_connection.close_all_connections()
