"""
Visualize global corpus topics from the Excel produced by
src.topic.export_global_topics.

Overview
--------
- Reads sheets: act_topics_detailed, topics_overall, topic_committee_overall, topics_by_year
- Builds a global topic co-occurrence network:
  - Nodes = topics
  - Node size = number of Acts mentioning the topic (acts_count)
  - Node color = dominant committee (optional)
  - Edges = co-occurrence in the same Act; edge weight = co-occur count
- Saves:
  - global_topics.png: co-occurrence network
  - top_topics.png: bar chart of top topics by acts_count
  - top_committees.png: bar chart of top committees by unique acts
  - topic_year_heatmap.png: heatmap for topic vs year (if year data present)

Usage
-----
python -m src.topic.visualize_global_topics \
  [--excel outputs/analytics/global_topics_YYYYMMDD_HHMMSS.xlsx] \
  [--labels] [--min-edge 2] [--min-degree 1] [--dpi 300] \
  [--label-top 50] [--top-nodes 100] [--output-dir outputs/analytics/global_topic_graphs]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd
import networkx as nx
import matplotlib
import os as _os

# Headless-safe defaults
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


def _find_latest_export(pattern: str = os.path.join("outputs", "analytics", "global_topics_*.xlsx")) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _load_sheets(excel_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(excel_path)
    detailed = pd.read_excel(xls, sheet_name="act_topics_detailed")
    topics_overall = pd.read_excel(xls, sheet_name="topics_overall")
    committee_topic = pd.read_excel(xls, sheet_name="topic_committee_overall")
    # topics_by_year may be empty placeholder
    try:
        topics_by_year = pd.read_excel(xls, sheet_name="topics_by_year")
    except Exception:
        topics_by_year = pd.DataFrame()
    return detailed, topics_overall, committee_topic, topics_by_year


def _cooccurrence_edges(detailed: pd.DataFrame) -> Counter[Tuple[str, str]]:
    det = detailed[["act_title", "topic"]].dropna()
    co = Counter()
    if det.empty:
        return co
    for act_title, grp in det.groupby("act_title"):
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
) -> str:
    G = nx.Graph()

    # node weights and metadata
    node_values: Dict[str, int] = {}
    for row in topic_stats.itertuples(index=False):
        topic = str(row.topic)
        value = int(row.acts_count) if hasattr(row, "acts_count") and pd.notnull(row.acts_count) else 1
        node_values[topic] = value
        meta: Dict[str, str] = {}
        if hasattr(row, "dominant_committee") and isinstance(row.dominant_committee, str):
            meta["committee"] = row.dominant_committee
        G.add_node(topic, **meta)

    for (a, b), w in co_occurs.items():
        if a not in node_values or b not in node_values:
            continue
        if int(w) < int(min_edge_weight):
            continue
        G.add_edge(str(a), str(b), weight=int(w))

    if not G.nodes:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No topics/edges to visualize", ha="center", va="center")
        ax.set_axis_off()
        fig.suptitle(title)
        fig.savefig(out_file, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return out_file

    # co-degree
    co_degree: Dict[str, int] = {n: 0 for n in G.nodes}
    for (a, b), w in co_occurs.items():
        if a in co_degree:
            co_degree[a] += int(w)
        if b in co_degree:
            co_degree[b] += int(w)

    if label_top is not None and label_top > 0:
        order = sorted(G.nodes(), key=lambda t: (node_values.get(t, 0), co_degree.get(t, 0), t), reverse=True)
        keep = set(order[: int(label_top)])
        G.remove_nodes_from([n for n in list(G.nodes) if n not in keep])

    n = G.number_of_nodes()
    iterations = 200 if n > 120 else 100
    k = 1.2 / (n ** 0.5) if n > 0 else None
    pos = nx.spring_layout(G, seed=42, iterations=iterations, k=k)

    sizes = [120 + 80 * (max(1, node_values.get(node, 1)) ** 0.5) for node in G.nodes]
    widths = [0.6 + 1.2 * (max(1, int(data.get("weight", 1))) ** 0.5) for _, _, data in G.edges(data=True)]

    if n <= 30:
        figsize = (12, 12)
    elif n <= 80:
        figsize = (14, 14)
    elif n <= 150:
        figsize = (16, 16)
    else:
        figsize = (18, 18)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_axis_off()

    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#B5B5B5", alpha=0.5)

    if color_by_committee:
        committees = [G.nodes[n].get("committee", "") for n in G.nodes]
        uniq = sorted({c for c in committees if c})
        from matplotlib import cm
        cmap = cm.get_cmap("tab20")
        color_map: Dict[str, str] = {c: matplotlib.colors.to_hex(cmap(i % 20)) for i, c in enumerate(uniq)}
        node_colors = [color_map.get(G.nodes[n].get("committee", ""), "#1f77b4") for n in G.nodes]
    else:
        node_colors = ["#1f77b4"] * n

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=node_colors, linewidths=0.5, edgecolors="#333333", alpha=0.9)

    if show_labels:
        label_limit = 30 if (label_top is None and n > 30) else (min(n, int(label_top)) if label_top else n)
        ordered_nodes = sorted(G.nodes(), key=lambda t: (node_values.get(t, 0), co_degree.get(t, 0), t), reverse=True)
        to_label = set(ordered_nodes[:label_limit])
        labels = {n: n for n in G.nodes if n in to_label}
        font_size = 11 if n <= 30 else (9 if n <= 100 else 7)
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=font_size)

    if color_by_committee:
        uniq = sorted({G.nodes[n].get("committee", "") for n in G.nodes if G.nodes[n].get("committee")})
        if uniq:
            from matplotlib.lines import Line2D
            from matplotlib import cm
            handles = []
            cmap = matplotlib.colormaps.get("tab20")
            for i, c in enumerate(uniq):
                handles.append(Line2D([0], [0], marker='o', color='w', label=c,
                                      markerfacecolor=matplotlib.colors.to_hex(cmap(i % 20)), markersize=8))
            ax.legend(handles=handles, title="Dominant Committee", loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.suptitle(title)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_file


def _bar_top_topics(topics_overall: pd.DataFrame, out_file: str, top_n: int = 30, dpi: int = 220) -> Optional[str]:
    if topics_overall.empty:
        return None
    df = topics_overall.copy()
    df = df.sort_values(["acts_count", "total_importance", "avg_importance", "topic"], ascending=[False, False, False, True]).head(int(top_n))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(6, int(0.35 * len(df)))))
    ax.barh(df["topic"].astype(str), df["acts_count"].astype(int), color="#4C72B0")
    ax.invert_yaxis()
    ax.set_title("Top topics by number of Acts")
    ax.set_xlabel("Acts count")
    fig.tight_layout()
    _ensure_dir(out_file)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _bar_top_committees(detailed: pd.DataFrame, out_file: str, top_n: int = 30, dpi: int = 220) -> Optional[str]:
    if detailed.empty:
        return None
    counts = (
        detailed.groupby("committee")["act_title"].nunique().sort_values(ascending=False).reset_index(name="acts_count")
    ).head(int(top_n))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(6, int(0.35 * len(counts)))))
    ax.barh(counts["committee"].astype(str), counts["acts_count"].astype(int), color="#55A868")
    ax.invert_yaxis()
    ax.set_title("Top committees by number of Acts")
    ax.set_xlabel("Acts count")
    fig.tight_layout()
    _ensure_dir(out_file)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _heatmap_topic_year(topics_year: pd.DataFrame, out_file: str, top_k_rows: int = 20, dpi: int = 220) -> Optional[str]:
    if topics_year is None or topics_year.empty:
        return None
    df = topics_year.copy()
    if "topic" not in df.columns or len(df.columns) <= 1:
        return None
    # Keep at most top_k_rows (already ordered in exporter), ensure years sorted
    df = df.head(int(top_k_rows))
    topics = df["topic"].astype(str).tolist()
    years = [c for c in df.columns if c != "topic"]
    data = df[years].astype(float).to_numpy()
    plt.style.use("seaborn-v0_8-whitegrid")
    width = max(8.0, min(14.0, 6.0 + 0.18 * len(years)))
    height = float(max(6, int(0.4 * len(topics))))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(data, aspect="auto", cmap="Blues")
    tick_indices = list(range(len(years)))
    if len(years) > 20:
        step = max(1, math.ceil(len(years) / 12))
        tick_indices = list(range(0, len(years), step))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([years[i] for i in tick_indices], rotation=45, ha="right")
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics)
    ax.set_title("Topic mentions by year (Acts count)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _ensure_dir(out_file)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file


def run(
    excel: Optional[str] = None,
    output_dir: str = os.path.join("outputs", "analytics", "global_topic_graphs"),
    labels: bool = True,
    dpi: int = 220,
    min_edge: int = 1,
    min_degree: int = 0,
    label_top: Optional[int] = 50,
    top_nodes: Optional[int] = None,
    color_by_committee: bool = True,
    top_bars: int = 30,
    top_heatmap_rows: int = 20,
) -> List[str]:
    excel_path = excel or _find_latest_export()
    if not excel_path or not os.path.exists(excel_path):
        raise FileNotFoundError("No global topics Excel found. Provide --excel or run export_global_topics first.")

    detailed, topics_overall, committee_topic, topics_year = _load_sheets(excel_path)

    # Build topic_stats from topics_overall
    topic_stats = topics_overall[["topic", "acts_count", "total_importance", "avg_importance", "dominant_committee"]].copy() if not topics_overall.empty else pd.DataFrame(columns=["topic", "acts_count"])  # type: ignore

    co = _cooccurrence_edges(detailed)

    # Optional filtering by min_degree using edge weights
    if min_degree > 0 and co:
        degrees = Counter()
        for (a, b), w in co.items():
            degrees[a] += int(w)
            degrees[b] += int(w)
        keep = {t for t, d in degrees.items() if d >= int(min_degree)}
        if not topic_stats.empty:
            topic_stats = topic_stats[topic_stats["topic"].astype(str).isin(keep)]
        co = Counter({(a, b): w for (a, b), w in co.items() if a in keep and b in keep})

    # Optional keep only top N nodes by acts_count then degree then name
    if top_nodes is not None and top_nodes > 0 and not topic_stats.empty:
        degrees = Counter()
        for (a, b), w in co.items():
            degrees[a] += int(w)
            degrees[b] += int(w)
        order = sorted(
            topic_stats["topic"].astype(str).tolist(),
            key=lambda t: (
                int(topic_stats.loc[topic_stats["topic"].astype(str) == t, "acts_count"].iloc[0]) if not topic_stats.empty else 0,
                int(degrees.get(t, 0)),
                t,
            ),
            reverse=True,
        )
        keep = set(order[: int(top_nodes)])
        topic_stats = topic_stats[topic_stats["topic"].astype(str).isin(keep)]
        co = Counter({(a, b): w for (a, b), w in co.items() if a in keep and b in keep})

    _ensure_dir(os.path.join(output_dir, "dummy"))
    saved: List[str] = []

    png_network = os.path.join(output_dir, "global_topics.png")
    _save_networkx_png(
        topic_stats,
        co,
        title="Global Topics Co-occurrence",
        out_file=png_network,
        show_labels=labels,
        dpi=int(dpi),
        min_edge_weight=int(min_edge),
        label_top=label_top,
        color_by_committee=color_by_committee,
    )
    saved.append(png_network)

    top_topics_png = os.path.join(output_dir, "top_topics.png")
    out = _bar_top_topics(topics_overall, top_topics_png, top_n=int(top_bars), dpi=int(dpi))
    if out:
        saved.append(out)

    top_committees_png = os.path.join(output_dir, "top_committees.png")
    out = _bar_top_committees(detailed, top_committees_png, top_n=int(top_bars), dpi=int(dpi))
    if out:
        saved.append(out)

    heatmap_png = os.path.join(output_dir, "topic_year_heatmap.png")
    out = _heatmap_topic_year(topics_year, heatmap_png, top_k_rows=int(top_heatmap_rows), dpi=int(dpi))
    if out:
        saved.append(out)

    return saved


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize global topics as PNG charts and network")
    ap.add_argument("--excel", type=str, default=None, help="Path to global_topics_*.xlsx (default: latest)")
    ap.add_argument("--labels", action="store_true", help="Draw topic labels on the network (default: off)")
    ap.add_argument("--min-edge", type=int, default=1, help="Hide edges with weight < N (default: 1)")
    ap.add_argument("--min-degree", type=int, default=0, help="Hide topics with co-occurrence degree < N (default: 0)")
    ap.add_argument("--label-top", type=int, default=50, help="Max number of nodes to label (default: 50)")
    ap.add_argument("--top-nodes", type=int, default=None, help="Restrict network to top N topics by occurrence")
    ap.add_argument("--no-committee-colors", action="store_true", help="Disable committee-based node colors")
    ap.add_argument("--dpi", type=int, default=220, help="PNG DPI (default: 220)")
    ap.add_argument("--top-bars", type=int, default=30, help="Top N bars for topic/committee charts (default: 30)")
    ap.add_argument("--top-heatmap-rows", type=int, default=20, help="Rows (topics) in heatmap (default: 20)")
    ap.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("outputs", "analytics", "global_topic_graphs"),
        help="Directory to write PNG files (default: outputs/analytics/global_topic_graphs)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = run(
        excel=args.excel,
        output_dir=args.output_dir,
        labels=bool(args.labels),
        dpi=int(args.dpi),
        min_edge=int(args.min_edge),
        min_degree=int(args.min_degree),
        label_top=int(args.label_top) if args.label_top else None,
        top_nodes=int(args.top_nodes) if args.top_nodes else None,
        color_by_committee=not args.no_committee_colors,
        top_bars=int(args.top_bars),
        top_heatmap_rows=int(args.top_heatmap_rows),
    )
    if paths:
        print("Saved:")
        for p in paths:
            print(f"  - {p}")
    else:
        print("No graphs were generated.")


if __name__ == "__main__":
    main()

