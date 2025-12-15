"""
Visualize community topics as static PNG graphs from the Excel produced by
src.topic.export_leiden_community_topics.

Overview
--------
- Reads the "act_topics_by_community" and "topics_by_community" sheets.
- Builds a topic co-occurrence network per community:
  - Nodes = topics (within a community)
  - Node size = number of Acts mentioning the topic in that community
  - Edges = co-occurrence in the same Act; edge weight = co-occur count
- Saves one PNG per community (no HTML output).

Usage
-----
python -m src.topic.visualize_community_topics \
  [--excel outputs/analytics/community_topics_YYYYMMDD_HHMMSS.xlsx] \
  [--communities 1 2 3 | --top 10 | --all] \
  [--labels] [--min-edge 2] [--min-degree 1] [--dpi 300] \
  [--output-dir outputs/analytics/community_topic_graphs]

Notes
-----
- If --excel is omitted, the script will pick the most recent
  outputs/analytics/community_topics_*.xlsx
- Automatically runs `src.topic.export_leiden_community_topics` before rendering when not doing a merge-only montage.
- Uses NetworkX + Matplotlib only.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Mapping, Any
import re

import pandas as pd
import networkx as nx
import matplotlib
import os as _os
from src.db.db_connection import db_connection as db_conn
from src.topic.export_leiden_community_topics import run as export_leiden_community_topics_run
# Ensure headless backend and writable MPL cache dir for CI/sandbox environments
if _os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
if "MPLCONFIGDIR" not in _os.environ:
    _os.makedirs(_os.path.join("outputs", "mplconfig"), exist_ok=True)
    _os.environ["MPLCONFIGDIR"] = _os.path.join("outputs", "mplconfig")
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _find_latest_export(pattern: str = os.path.join("outputs", "analytics", "community_topics_*.xlsx")) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _load_sheets(excel_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (detailed, aggregated) DataFrames from the Excel workbook.

    - detailed: sheet "act_topics_by_community"
    - aggregated: sheet "topics_by_community"
    """
    xls = pd.ExcelFile(excel_path)
    detailed = pd.read_excel(xls, sheet_name="act_topics_by_community")
    aggregated = pd.read_excel(xls, sheet_name="topics_by_community")
    return detailed, aggregated


def _community_size_map_from_aggregated(aggregated: pd.DataFrame) -> Dict[int, int]:
    if aggregated.empty:
        return {}
    sizes = (
        aggregated.groupby("community")
        .agg(community_size=("community_size", "max"))
        .reset_index()
    )
    return {
        int(row.community): int(row.community_size)
        for row in sizes.itertuples(index=False)
        if pd.notnull(row.community) and pd.notnull(row.community_size)
    }


def _community_size_map_from_excel(excel_path: Optional[str]) -> Dict[int, int]:
    path = excel_path or _find_latest_export()
    if not path or not os.path.exists(path):
        return {}
    try:
        _, aggregated = _load_sheets(path)
    except Exception:
        return {}
    return _community_size_map_from_aggregated(aggregated)


def _community_id_from_path(path: str) -> Optional[int]:
    m = re.search(r"community_(\d+)_topics\.png$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def _topic_stats_for_community(aggregated: pd.DataFrame, community: int) -> pd.DataFrame:
    """Compute per-topic stats within a community using the aggregated sheet.

    Returns columns:
      - topic, acts_count, total_importance, committees (comma-separated),
        dominant_committee (committee with highest acts_count for this topic)
    """
    agg_c = aggregated[aggregated["community"] == community].copy()
    if agg_c.empty:
        return pd.DataFrame(
            columns=[
                "topic",
                "acts_count",
                "total_importance",
                "committees",
                "dominant_committee",
            ]
        )

    # Per-topic totals
    topic_totals = (
        agg_c.groupby("topic").agg(
            acts_count=("acts_count", "sum"),
            total_importance=("total_importance", "sum"),
            committees=("committee", lambda s: ", ".join(sorted(set(map(str, s)))))
        )
    ).reset_index()

    # Dominant committee per topic (by highest acts_count under that committee)
    tc = (
        agg_c.groupby(["topic", "committee"]).agg(acts_count=("acts_count", "sum")).reset_index()
    )
    # Select committee with max acts_count for each topic (tie-broken alphabetically)
    tc_sorted = tc.sort_values(["topic", "acts_count", "committee"], ascending=[True, False, True])
    dominant = tc_sorted.groupby("topic").first().reset_index()[["topic", "committee"]]
    dominant.columns = ["topic", "dominant_committee"]

    out = topic_totals.merge(dominant, on="topic", how="left")
    return out


def _committee_dominance_stats(detailed: pd.DataFrame, community: int) -> List[Dict[str, Any]]:
    """Return ordered committee dominance stats for a community."""
    det = detailed[detailed["community"] == community][["committee", "act_title", "community_size"]].dropna(
        subset=["committee", "act_title"]
    )
    if det.empty:
        return []

    # Determine community size for percentage calculations
    size_val = det["community_size"].dropna()
    community_size = int(size_val.iloc[0]) if not size_val.empty else None

    counts = (
        det.groupby("committee")["act_title"]
        .nunique()
        .reset_index(name="acts_count")
        .sort_values(["acts_count", "committee"], ascending=[False, True])
    )

    stats: List[Dict[str, Any]] = []
    for row in counts.itertuples(index=False):
        committee = str(row.committee)
        if not committee:
            continue
        pct = None
        if community_size and community_size > 0:
            pct = (int(row.acts_count) / community_size) * 100.0
        stats.append({"committee": committee, "acts": int(row.acts_count), "pct": pct})
    return stats


def _cooccurrence_edges(detailed: pd.DataFrame, community: int) -> Counter[Tuple[str, str]]:
    """From per-act topic rows within a community, compute co-occurrence counts.

    For each Act in the community, take the set of topics assigned to the Act
    (deduped across committees) and add +1 to every unordered pair.
    """
    det_c = detailed[detailed["community"] == community][["act_title", "topic"]].dropna()
    co = Counter()
    if det_c.empty:
        return co
    for act_title, grp in det_c.groupby("act_title"):
        topics = sorted(set(map(str, grp["topic"].tolist())))
        if len(topics) < 2:
            continue
        for a, b in combinations(topics, 2):
            co[(a, b)] += 1
    return co


def _save_networkx_png(
    topic_stats: pd.DataFrame,
    co_occurs: Counter[Tuple[str, str]],
    *,
    title: str,
    out_file: str,
    show_labels: bool = False,
    dpi: int = 220,
    min_edge_weight: int = 1,
    label_top: Optional[int] = None,
    color_by_committee: bool = True,
    committee_stats: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Render a static PNG using NetworkX + Matplotlib and return the file path.

    - Node size scaled by acts_count (sqrt scaling)
    - Edge width scaled by co-occurrence count (sqrt scaling)
    - Optional labels (off by default to reduce clutter)
    """
    G = nx.Graph()
    committee_stats_list: List[Mapping[str, Any]] = list(committee_stats) if committee_stats else []

    # Add nodes with weights
    node_values = {}
    committee_strength: Counter[str] = Counter()
    for row in topic_stats.itertuples(index=False):
        topic = str(row.topic)
        value = int(row.acts_count) if pd.notnull(row.acts_count) else 1
        node_values[topic] = value
        meta: Dict[str, str] = {}
        if hasattr(row, "dominant_committee") and isinstance(row.dominant_committee, str) and row.dominant_committee:
            committee = row.dominant_committee
            meta["committee"] = committee
            committee_strength[committee] += value
        G.add_node(topic, **meta)

    # Add edges (filter by threshold)
    for (a, b), w in co_occurs.items():
        if a not in node_values or b not in node_values:
            continue
        if int(w) < int(min_edge_weight):
            continue
        G.add_edge(str(a), str(b), weight=int(w))

    # Handle empty graph
    if not G.nodes:
        # Ensure directory exists and write a tiny placeholder figure
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No topics/edges to visualize", ha="center", va="center")
        ax.set_axis_off()
        fig.suptitle(title)
        fig.savefig(out_file, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return out_file

    # Compute co-occurrence degree (sum of weights incident to node)
    co_degree: Dict[str, int] = {n: 0 for n in G.nodes}
    for (a, b), w in co_occurs.items():
        if a in co_degree:
            co_degree[a] += int(w)
        if b in co_degree:
            co_degree[b] += int(w)

    # Optional node filtering by top N topics (by acts_count, tie-breaker co_degree)
    if label_top is not None and label_top > 0:
        order = sorted(
            G.nodes(),
            key=lambda t: (node_values.get(t, 0), co_degree.get(t, 0), t),
            reverse=True,
        )
        keep = set(order[: int(label_top)])
        # Remove nodes not in top list
        to_remove = [n for n in G.nodes if n not in keep]
        G.remove_nodes_from(to_remove)
        # Recompute n after pruning
    n = G.number_of_nodes()
    # Layout parameters scale with n for readability
    iterations = 200 if n > 120 else 100
    k = 1.2 / (n**0.5) if n > 0 else None
    pos = nx.spring_layout(G, seed=42, iterations=iterations, k=k)

    # Node sizes (sqrt scaling): base 120, grow with sqrt(value)
    sizes = []
    for node in G.nodes:
        v = max(1, node_values.get(node, 1))
        sizes.append(120 + 80 * (v ** 0.5))

    # Edge widths (sqrt scaling)
    widths = []
    for u, v, data in G.edges(data=True):
        w = max(1, int(data.get("weight", 1)))
        widths.append(0.6 + 1.2 * (w ** 0.5))

    # Figure size heuristic
    if n <= 30:
        figsize = (10, 10)
    elif n <= 80:
        figsize = (12, 12)
    elif n <= 150:
        figsize = (14, 14)
    else:
        figsize = (16, 16)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_axis_off()

    # Draw edges first
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#B5B5B5", alpha=0.5)

    # Node colors by dominant committee (optional)
    if color_by_committee:
        ordered_committees: List[str] = []
        stats_map: Dict[str, Mapping[str, Any]] = {}
        if committee_stats_list:
            for entry in committee_stats_list:
                committee_name = str(entry.get("committee", "")).strip()
                if not committee_name:
                    continue
                stats_map[committee_name] = entry
                ordered_committees.append(committee_name)
        else:
            ordered_committees = [
                c
                for c, _ in sorted(
                    committee_strength.items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
                if c
            ]
        cmap = matplotlib.colormaps["tab20"]
        color_map: Dict[str, str] = {
            c: matplotlib.colors.to_hex(cmap(i % cmap.N)) for i, c in enumerate(ordered_committees)
        }
        node_colors = [
            color_map.get(G.nodes[n].get("committee", ""), "#1f77b4") for n in G.nodes
        ]
    else:
        node_colors = ["#1f77b4"] * n

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=node_colors,
        linewidths=0.5,
        edgecolors="#333333",
        alpha=0.9,
    )

    # Optional labels
    if show_labels:
        # Label at most 'label_top' nodes when provided; else label all (small graphs) or top 30
        if label_top is None:
            label_limit = 30 if n > 30 else n
        else:
            label_limit = min(n, int(label_top))
        ordered_nodes = sorted(
            G.nodes(),
            key=lambda t: (node_values.get(t, 0), co_degree.get(t, 0), t),
            reverse=True,
        )
        to_label = set(ordered_nodes[:label_limit])
        labels = {n: n for n in G.nodes if n in to_label}

        # Smaller font for larger graphs
        if n <= 30:
            font_size = 11
        elif n <= 100:
            font_size = 9
        else:
            font_size = 7
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=font_size)

    # Draw legend for committees if colored
    if color_by_committee:
        uniq = ordered_committees if ordered_committees else []
        if uniq:
            from matplotlib.lines import Line2D
            handles = []
            cmap = matplotlib.colormaps.get("tab20")
            for i, c in enumerate(uniq):
                entry = stats_map.get(c) if committee_stats_list else None
                if entry:
                    acts = entry.get("acts")
                    pct = entry.get("pct")
                    if acts is not None and pct is not None:
                        label_text = f"{c} ({acts} Acts, {pct:.0f}%)"
                    elif acts is not None:
                        label_text = f"{c} ({acts} Acts)"
                    else:
                        label_text = c
                else:
                    strength = committee_strength.get(c, 0)
                    label_text = f"{c} ({strength} topic mentions)" if strength else c
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=label_text,
                        markerfacecolor=matplotlib.colors.to_hex(cmap(i % 20)),
                        markersize=8,
                    )
                )
            ax.legend(handles=handles, title="Dominant Committee", loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.suptitle(title)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_file


def _create_montage(
    image_paths: List[str],
    output_filename: str,
    cols: int = 2,
    dpi: int = 220,
    *,
    size_map: Optional[Mapping[int, int]] = None,
) -> Optional[str]:
    """Create a grid montage from a list of PNGs and save to output_filename.

    - Uses matplotlib only; no extra dependencies.
    - Titles are inferred from filenames (e.g., community_5_topics.png -> "Community 5").
    """
    if not image_paths:
        return None

    import math

    # Filter to existing files
    paths = [p for p in image_paths if os.path.exists(p)]
    if not paths:
        return None

    def _path_sort_key(p: str) -> Tuple[int, int, str]:
        comm_id = _community_id_from_path(p)
        if size_map and comm_id is not None:
            return (-size_map.get(comm_id, 0), comm_id, p)
        if comm_id is None:
            return (10**9, 10**9, p)
        return (0, comm_id, p)

    paths.sort(key=_path_sort_key)

    n = len(paths)
    cols = max(1, min(cols, n))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.0, rows * 5.2), dpi=dpi)
    if isinstance(axes, (list, tuple)):
        pass
    import numpy as np
    axes = np.array(axes).reshape(-1)

    for idx, p in enumerate(paths):
        ax = axes[idx]
        try:
            img = plt.imread(p)
            ax.imshow(img)
            ax.set_axis_off()
            base = os.path.basename(p)
            if base.startswith("community_") and base.endswith("_topics.png"):
                try:
                    comm_id = int(base.split("_")[1])
                    title = f"Community {comm_id}"
                except Exception:
                    title = base
            else:
                title = base
            ax.set_title(title, fontsize=12, pad=6)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{os.path.basename(p)}\n{e}", ha='center', va='center', fontsize=8)
            ax.set_axis_off()

    for ax in axes[n:]:
        ax.set_visible(False)

    try:
        fig.tight_layout(pad=0.2)
    except Exception:
        pass
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fig.savefig(output_filename, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved merged montage to {output_filename}")
    return output_filename


def _find_existing_community_pngs(output_dir: str) -> List[str]:
    """Return list of existing community_XX_topics.png files sorted by community id."""
    pattern = os.path.join(output_dir, "community_*_topics.png")
    paths = [p for p in glob.glob(pattern) if os.path.exists(p)]
    # Sort by community id where possible
    def _community_sort_key(p: str) -> Tuple[int, str]:
        m = re.search(r"community_(\d+)_topics", os.path.basename(p))
        return (int(m.group(1)) if m else 10**9, p)
    return sorted(paths, key=_community_sort_key)


def render_community_graph(
    excel_path: str,
    community: int,
    output_dir: str,
    *,
    min_degree: int = 0,
    labels: bool = False,
    dpi: int = 220,
    min_edge: int = 1,
    label_top: Optional[int] = None,
    top_nodes: Optional[int] = None,
    color_by_committee: bool = True,
) -> List[str]:
    """Render one community's topic graph as PNG and return [path]."""
    detailed, aggregated = _load_sheets(excel_path)

    topic_stats = _topic_stats_for_community(aggregated, community)
    if topic_stats.empty:
        return []

    co = _cooccurrence_edges(detailed, community)

    # Optionally filter nodes by degree threshold using the computed edges
    if min_degree > 0 and co:
        degrees = Counter()
        for (a, b), w in co.items():
            degrees[a] += int(w)
            degrees[b] += int(w)
        keep_topics = {t for t, d in degrees.items() if d >= min_degree}
        topic_stats = topic_stats[topic_stats["topic"].astype(str).isin(keep_topics)]
        # Drop edges involving filtered nodes
        co = Counter({(a, b): w for (a, b), w in co.items() if a in keep_topics and b in keep_topics})

    # Optional: restrict to top N nodes by acts_count (tie-break co-degree)
    if top_nodes is not None and top_nodes > 0:
        # Build degree scores
        degrees = Counter()
        for (a, b), w in co.items():
            degrees[a] += int(w)
            degrees[b] += int(w)
        # Order topics by acts_count then degree then name
        order = sorted(
            topic_stats["topic"].astype(str).tolist(),
            key=lambda t: (
                int(topic_stats.loc[topic_stats["topic"].astype(str) == t, "acts_count"].iloc[0]),
                int(degrees.get(t, 0)),
                t,
            ),
            reverse=True,
        )
        keep = set(order[: int(top_nodes)])
        topic_stats = topic_stats[topic_stats["topic"].astype(str).isin(keep)]
        co = Counter({(a, b): w for (a, b), w in co.items() if a in keep and b in keep})

    _ensure_dir(os.path.join(output_dir, "dummy"))
    png_file = os.path.join(output_dir, f"community_{community}_topics.png")
    _save_networkx_png(
        topic_stats,
        co,
        title=f"Community {community} Topics",
        out_file=png_file,
        show_labels=labels,
        dpi=int(dpi),
        min_edge_weight=int(min_edge),
        label_top=label_top,
        color_by_committee=bool(color_by_committee),
        committee_stats=_committee_dominance_stats(detailed, community),
    )
    return [png_file]


def run(
    excel: Optional[str] = None,
    communities: Optional[Sequence[int]] = None,
    top: Optional[int] = 10,
    output_dir: str = os.path.join("outputs", "analytics", "community_topic_graphs"),
    min_degree: int = 0,
    labels: bool = True,
    dpi: int = 220,
    min_edge: int = 1,
    label_top: Optional[int] = 30,
    top_nodes: Optional[int] = None,
    color_by_committee: bool = True,
    merge: bool = False,
    merge_columns: int = 2,
    merge_existing: bool = False,
    merge_only: bool = False,
) -> List[str]:
    """Render graphs for selected communities; optionally merge into a montage.

    - If communities is provided, visualize exactly those IDs.
    - Else, picks top N communities by size (descending) using the aggregated sheet.
    - If excel is None, uses the most recent community_topics_*.xlsx.
    """
    # Merge-only: skip rendering, just combine existing PNGs in output_dir
    if merge_only:
        size_map = _community_size_map_from_excel(excel)
        existing = _find_existing_community_pngs(output_dir)
        if not existing:
            print("No existing community PNGs found to merge.")
            return []
        montage_path = os.path.join(output_dir, "communities_merged.png")
        _create_montage(
            existing,
            montage_path,
            cols=max(1, int(merge_columns)),
            dpi=dpi,
            size_map=size_map,
        )
        return existing + [montage_path]

    exported_excel_path: Optional[str] = None
    try:
        exported_excel_path = export_leiden_community_topics_run()
    finally:
        db_conn.close_all_connections()

    excel_path = excel or exported_excel_path or _find_latest_export()
    if not excel_path or not os.path.exists(excel_path):
        raise FileNotFoundError(
            "No community topics Excel found. Provide --excel or run export_leiden_community_topics first."
        )

    detailed, aggregated = _load_sheets(excel_path)
    if aggregated.empty:
        return []

    community_size_map = _community_size_map_from_aggregated(aggregated)

    if communities:
        comm_ids = list({int(c) for c in communities})
    else:
        # Choose top by community_size (desc)
        sizes = (
            aggregated.groupby("community").agg(community_size=("community_size", "max")).reset_index()
        )
        sizes = sizes.sort_values(["community_size", "community"], ascending=[False, True])
        if top is None or top <= 0:
            comm_ids = sizes["community"].astype(int).tolist()
        else:
            comm_ids = sizes.head(int(top))["community"].astype(int).tolist()

    saved: List[str] = []
    for comm_id in comm_ids:
        outs = render_community_graph(
            excel_path,
            int(comm_id),
            output_dir,
            min_degree=min_degree,
            labels=labels,
            dpi=dpi,
            min_edge=min_edge,
            label_top=label_top,
            top_nodes=top_nodes,
            color_by_committee=color_by_committee,
        )
        saved.extend(outs)

    # Optionally create a merged montage image in N columns
    if merge:
        montage_path = os.path.join(output_dir, "communities_merged.png")
        try:
            if merge_existing:
                candidates = _find_existing_community_pngs(output_dir)
            else:
                candidates = saved
            if candidates:
                _create_montage(
                    candidates,
                    montage_path,
                    cols=max(1, int(merge_columns)),
                    dpi=dpi,
                    size_map=community_size_map,
                )
                saved.append(montage_path)
            else:
                print("Merge requested but no PNGs found.")
        except Exception as e:
            print(f"Warning: failed to create montage: {e}")
    return saved


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize community topics as static PNG graphs")
    ap.add_argument("--excel", type=str, default=None, help="Path to community_topics_*.xlsx (default: latest)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--communities", type=int, nargs="+", help="Specific community ID(s) to visualize")
    grp.add_argument("--top", type=int, default=10, help="Top N communities by size (default: 10)")
    grp.add_argument("--all", action="store_true", help="Visualize all communities")
    ap.add_argument("--min-degree", type=int, default=0, help="Hide topics with degree < N (default: 0)")
    ap.add_argument("--min-edge", type=int, default=1, help="Hide edges with weight < N (default: 1)")
    ap.add_argument("--no-labels", action="store_true", help="Do not draw labels (default: labels shown)")
    ap.add_argument("--label-top", type=int, default=30, help="Max number of nodes to label (default: 30)")
    ap.add_argument("--top-nodes", type=int, default=None, help="Restrict to top N topics by occurrence")
    ap.add_argument("--no-committee-colors", action="store_true", help="Disable committee-based node colors")
    ap.add_argument("--dpi", type=int, default=220, help="PNG DPI (default: 220)")
    ap.add_argument("--merge", action="store_true", help="Also save a combined montage of all PNGs")
    ap.add_argument("--merge-columns", type=int, default=2, help="Number of columns in the merged montage (default: 2)")
    ap.add_argument("--merge-existing", action="store_true", help="Merge all existing community_*.png in the output directory")
    ap.add_argument("--merge-only", action="store_true", help="Only create the merged montage from existing PNGs (no rendering)")
    ap.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("outputs", "analytics", "community_topic_graphs"),
        help="Directory to write PNG files (default: outputs/analytics/community_topic_graphs)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    top = None if args.all else args.top
    try:
        paths = run(
            excel=args.excel,
            communities=args.communities,
            top=top,
            output_dir=args.output_dir,
            min_degree=args.min_degree,
            labels=not args.no_labels,
            dpi=args.dpi,
            min_edge=args.min_edge,
            label_top=args.label_top,
            top_nodes=args.top_nodes,
            color_by_committee=not args.no_committee_colors,
            merge=args.merge,
            merge_columns=args.merge_columns,
            merge_existing=args.merge_existing,
            merge_only=args.merge_only,
        )
    finally:
        # No persistent connections to close here; keep symmetry with other modules
        pass

    if paths:
        print("Saved:")
        for p in paths:
            print(f"  - {p}")
    else:
        print("No graphs were generated.")


if __name__ == "__main__":
    main()
