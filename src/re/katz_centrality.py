"""
Katz centrality over act relationships (directed, weighted, JSONB-aware).

Overview
--------
Katz centrality measures the influence of a node by summing the number of
walks that end at that node, with longer walks exponentially attenuated.
Formally, for a directed graph with adjacency matrix A, the (in-)Katz
centrality c solves:

    c = alpha * A^T * c + beta * 1

where alpha is an attenuation factor and beta is a positive constant that
ensures nodes with no in-edges still receive a baseline score. The solution
exists and the fixed-point iteration converges when alpha < 1 / lambda_max,
with lambda_max the spectral radius of A (equivalently A^T).

This program builds A from the PostgreSQL table `act_relationships`, where
each row represents a directed edge subject_name -> object_name and the
`relationships` JSONB field stores distinct relationship types between the
pair. Multiple relationship types for the same pair are important for Katz:
they increase the number of potential walks, so we support weighted edges:

    - unweighted:         A_ij = 1 if any relationship exists
    - multiplicity:       A_ij = number of relationship types (default)
    - log-multiplicity:   A_ij = 1 + log(1 + number of types)

Directionality is preserved (subject -> object). Self-loops are ignored.
We estimate lambda_max via power iteration on A and set alpha to 0.9 / lambda
by default. You can override alpha and/or beta via CLI flags.

Output
------
Writes CSV to `outputs/re/centrality/` with columns:

    Act title, Katz centrality

sorted by descending centrality. The filename includes a timestamp.

Notes
-----
- Heavier weights increase lambda_max; we re-estimate if the weighting changes.
- For undirected influence, pass --symmetrize to use A + A^T (then iterate with
  that symmetric adjacency and the same fixed-point scheme).
- The implementation uses adjacency lists and iterative multiplications, which
  scales to mid-sized graphs without external libraries.
"""

import argparse
import csv
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple

import networkx as nx

from src.db.db_connection import db_connection


def fetch_edges(weight_scheme: str = "multiplicity") -> Tuple[List[str], Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
    """Load nodes and weighted adjacency from the database.

    Returns:
        nodes: index -> act title (list where position is node id)
        out_adj: dict[node] -> list of (neighbor, weight)
        in_adj: dict[node] -> list of (predecessor, weight)
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
                # rels is JSONB array -> Python list of strings
                rels_list = rels or []
                edges_raw.append((s, o, rels_list))
    finally:
        db_connection.release_connection(conn)

    # Nodes: union of subjects and objects
    titles = sorted(set(subjects) | set(objects))
    idx_of: Dict[str, int] = {t: i for i, t in enumerate(titles)}

    out_adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(titles))}
    in_adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(titles))}

    for s, o, rels_list in edges_raw:
        if s == o:
            continue  # ignore self-loops
        i = idx_of[s]
        j = idx_of[o]

        m = max(1, len(rels_list))  # at least 1 if any edge exists
        if weight_scheme == "unweighted":
            w = 1.0
        elif weight_scheme == "log":
            w = 1.0 + math.log(1.0 + m)
        else:  # multiplicity (default)
            w = float(m)

        out_adj[i].append((j, w))
        in_adj[j].append((i, w))

    return titles, out_adj, in_adj


def symmetrize(out_adj: Dict[int, List[Tuple[int, float]]], in_adj: Dict[int, List[Tuple[int, float]]]) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
    """Return symmetric adjacency lists (A <- A + A^T)."""
    n = len(out_adj)
    out_sym: Dict[int, List[Tuple[int, float]]] = {i: list(neigh) for i, neigh in out_adj.items()}
    in_sym: Dict[int, List[Tuple[int, float]]] = {i: list(neigh) for i, neigh in in_adj.items()}

    # For each i->j in original, ensure j->i exists with accumulated weights
    for i, neigh in out_adj.items():
        for j, w in neigh:
            out_sym[j].append((i, w))
            in_sym[i].append((j, w))

    return out_sym, in_sym


def power_iteration_lambda(out_adj: Dict[int, List[Tuple[int, float]]], max_iter: int = 100, tol: float = 1e-9) -> float:
    """Estimate spectral radius lambda_max of A using power iteration.

    We iterate x <- A x with L1 normalization and return the stabilized
    scaling factor as an estimate of lambda_max. For nonnegative A, this is
    a practical estimator.
    """
    n = len(out_adj)
    if n == 0:
        return 0.0

    x = [1.0 / n] * n
    lam = 0.0

    for _ in range(max_iter):
        y = [0.0] * n
        for i, neigh in out_adj.items():
            xi = x[i]
            if xi == 0.0:
                continue
            for j, w in neigh:
                y[j] += w * xi

        norm1 = sum(abs(v) for v in y)
        if norm1 == 0.0:
            return 0.0
        y = [v / norm1 for v in y]

        # Rayleigh-like estimate via norm ratio on L1
        lam_new = norm1 / max(sum(abs(v) for v in x), 1e-12)
        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = lam_new
        x = y

    return lam


def katz_centrality(
    in_adj: Dict[int, List[Tuple[int, float]]],
    alpha: float,
    beta: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> List[float]:
    """Fixed-point iteration for c = alpha * A^T * c + beta * 1.

    Uses in-adjacency so each node i aggregates from its predecessors k:
        c_{t+1}[i] = alpha * sum_k w_{k,i} * c_t[k] + beta
    """
    n = len(in_adj)
    c = [1.0] * n  # start with baseline

    for _ in range(max_iter):
        c_next = [beta] * n
        for i in range(n):
            total = 0.0
            for k, w in in_adj[i]:
                total += w * c[k]
            c_next[i] += alpha * total

        # Check convergence (L1)
        diff = sum(abs(c_next[i] - c[i]) for i in range(n))
        c = c_next
        if diff < tol:
            break

    return c


def write_csv(path: str, rows: List[Tuple[str, float, str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Act title", "Katz centrality", "Explanation"])
        for t, score, expl in rows:
            w.writerow([t, f"{score:.10f}", expl])


def main():
    ap = argparse.ArgumentParser(description="Compute Katz centrality from act_relationships and export CSV.")
    ap.add_argument("--weight", choices=["multiplicity", "unweighted", "log"], default="multiplicity", help="Edge weight scheme.")
    ap.add_argument("--alpha", type=float, default=None, help="Attenuation factor. If omitted, auto-set to 0.9 / lambda_max.")
    ap.add_argument("--beta", type=float, default=1.0, help="Baseline additive term beta.")
    ap.add_argument("--symmetrize", action="store_true", help="Use A + A^T (undirected influence).")
    ap.add_argument("--max-iter", type=int, default=1000, help="Max iterations for Katz and power iteration.")
    ap.add_argument("--tol", type=float, default=1e-9, help="Convergence tolerance.")
    args = ap.parse_args()

    # Build graph directly from DB and compute via NetworkX
    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("PostgreSQL connection not available.")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT subject_name, object_name, relationships
                FROM act_relationships
                ORDER BY subject_name, object_name
                """
            )
            rows_db = cur.fetchall()
    finally:
        db_connection.release_connection(conn)

    G = nx.DiGraph()
    for s, o, rels in rows_db:
        if s == o:
            continue  # ignore self-loops
        rels_list = rels or []
        multiplicity = max(1, len(rels_list))
        if args.weight == "unweighted":
            w = 1.0
        elif args.weight == "log":
            w = 1.0 + math.log(1.0 + multiplicity)
        else:
            w = float(multiplicity)
        if G.has_edge(s, o):
            G[s][o]["weight"] += w
        else:
            G.add_edge(s, o, weight=w)

    if args.symmetrize:
        G = G.to_undirected(reciprocal=False)

    alpha = args.alpha if args.alpha is not None else 0.01
    scores = nx.katz_centrality(
        G,
        alpha=alpha,
        beta=args.beta,
        max_iter=args.max_iter,
        tol=args.tol,
        weight="weight",
        normalized=False,
    )

    # Build explanation strings per node: beta plus top neighbor contributions
    def top_contributors(node: str, top_k: int = 3) -> List[Tuple[str, float, float]]:
        contribs: List[Tuple[str, float, float]] = []  # (neighbor, contrib, weight)
        if G.is_directed():
            edges_iter = G.in_edges(node, data=True)
        else:
            edges_iter = ((nbr, node, G[nbr][node]) for nbr in G.neighbors(node))
        for src, _, data in edges_iter:
            w = float(data.get("weight", 1.0))
            c_src = float(scores.get(src, 0.0))
            contrib = alpha * w * c_src
            if contrib > 0.0:
                contribs.append((src, contrib, w))
        contribs.sort(key=lambda x: x[1], reverse=True)
        return contribs[:top_k]

    rows: List[Tuple[str, float, str]] = []
    for node, score in scores.items():
        parts: List[str] = []
        parts.append(f"beta={args.beta:.3g}")
        tc = top_contributors(node)
        if tc:
            tc_str = ", ".join(f"{nbr}({contrib:.3g},w={int(w) if abs(w - int(w)) < 1e-9 else w:.3g})" for nbr, contrib, w in tc)
            parts.append(f"top: {tc_str}")
        expl = "; ".join(parts)
        rows.append((node, score, expl))

    rows.sort(key=lambda x: (-x[1], x[0]))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", "re", "centrality")
    out_path = os.path.join(out_dir, f"katz_centrality_{args.weight}_{ts}.csv")
    write_csv(out_path, rows)

    print(f"Saved Katz centrality to: {out_path}")
    print(f"alpha={alpha}, beta={args.beta}, symmetrize={args.symmetrize}, weight={args.weight}")


if __name__ == "__main__":
    try:
        main()
    finally:
        db_connection.close_all_connections()
