"""
Compute high-level Neo4j graph stats for legislation network.

Reports:
- Total distinct Act nodes
- Act nodes by status (e.g., enacted vs historical)
- Total edges between Act nodes
- Edges grouped by relationship type
- Acts-by-year histogram saved to outputs/analytics/acts_by_year.png

Usage:
  poetry run python src/analytics/neo4j_graph_stats.py [--database DB] [--acts-by-year-output PATH] [--year-numbers] [--smooth N]

This will always generate three images by default:
- Bar chart: acts_by_year.png (or the provided output path)
- Line chart: acts_by_year_line.png
- Area chart: acts_by_year_area.png

Additionally, always writes an Excel workbook to outputs/analytics/acts_by_year.xlsx
containing a single sheet "ByYear": columns are years and each column lists
the Act titles for that year down the rows (skips year 0).
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple
import os
import matplotlib.pyplot as plt
import pandas as pd

from src.graph.neo4j_connection import neo4j_session, close_driver


def fetch_total_nodes(session) -> int:
    record = session.run("MATCH (a:Act) RETURN count(a) AS cnt").single()
    return int(record["cnt"]) if record and record["cnt"] is not None else 0


def fetch_nodes_by_status(session) -> List[Tuple[str, int]]:
    result = session.run(
        """
        MATCH (a:Act)
        RETURN coalesce(a.status, '(missing)') AS status, count(a) AS cnt
        ORDER BY cnt DESC, status ASC
        """
    )
    rows: List[Tuple[str, int]] = []
    for rec in result:
        rows.append((rec["status"], int(rec["cnt"])) )
    return rows


def fetch_total_edges(session) -> int:
    record = session.run(
        "MATCH (:Act)-[r]->(:Act) RETURN count(r) AS cnt"
    ).single()
    return int(record["cnt"]) if record and record["cnt"] is not None else 0


def fetch_edges_by_type(session) -> List[Tuple[str, int]]:
    result = session.run(
        """
        MATCH (:Act)-[r]->(:Act)
        RETURN type(r) AS rel_type, count(r) AS cnt
        ORDER BY cnt DESC, rel_type ASC
        """
    )
    rows: List[Tuple[str, int]] = []
    for rec in result:
        rows.append((rec["rel_type"], int(rec["cnt"])) )
    return rows


def fetch_acts_by_year(session) -> List[Tuple[int, int]]:
    """Return list of (year, count) for Act nodes with a year property."""
    result = session.run(
        """
        MATCH (a:Act)
        WHERE a.year IS NOT NULL
        RETURN toInteger(a.year) AS year, count(a) AS cnt
        ORDER BY year ASC
        """
    )
    rows: List[Tuple[int, int]] = []
    for rec in result:
        try:
            rows.append((int(rec["year"]), int(rec["cnt"])) )
        except Exception:
            # Skip malformed entries defensively
            continue
    return rows


def fetch_act_titles_by_year(session) -> List[Tuple[int, str]]:
    """Return list of (year, title) for Act nodes, skipping year 0 and NULLs."""
    result = session.run(
        """
        MATCH (a:Act)
        WHERE a.year IS NOT NULL AND toInteger(a.year) <> 0
        RETURN toInteger(a.year) AS year, a.title AS title
        ORDER BY year ASC, title ASC
        """
    )
    rows: List[Tuple[int, str]] = []
    for rec in result:
        try:
            y = int(rec["year"]) if rec["year"] is not None else None
            t = str(rec["title"]).strip() if rec["title"] is not None else None
            if y is None or y == 0 or not t:
                continue
            rows.append((y, t))
        except Exception:
            continue
    return rows


def _moving_average_centered(values: List[float], window: int) -> List[float]:
    """Return centered moving average with edge-padding.

    - If window < 2: returns values unchanged.
    - If window is even: uses window+1 to keep it centered.
    - Pads with edge values so the output length matches input length.
    """
    n = len(values)
    if n == 0 or window < 2:
        return values[:]
    if window % 2 == 0:
        window += 1
    k = window // 2
    # Pad with edge values
    padded: List[float] = [values[0]] * k + values[:] + [values[-1]] * k
    # Sliding window sum
    out: List[float] = []
    s = sum(padded[0:window])
    out.append(s / window)
    for i in range(1, n):
        s += padded[i + window - 1] - padded[i - 1]
        out.append(s / window)
    return out


def compute_stats(*, database: Optional[str] = None) -> Dict[str, object]:
    with neo4j_session(database=database) as session:
        total_nodes = fetch_total_nodes(session)
        nodes_by_status = fetch_nodes_by_status(session)
        total_edges = fetch_total_edges(session)
        edges_by_type = fetch_edges_by_type(session)

    return {
        "total_nodes": total_nodes,
        "nodes_by_status": nodes_by_status,
        "total_edges": total_edges,
        "edges_by_type": edges_by_type,
        "distinct_relationship_types": len(edges_by_type),
    }


def print_stats(stats: Dict[str, object]) -> None:
    print("Neo4j Graph Stats (Acts and relationships)")
    print("-")
    print(f"Total Act nodes: {stats['total_nodes']}")

    print("Nodes by status:")
    nodes_by_status: List[Tuple[str, int]] = stats["nodes_by_status"]  # type: ignore[assignment]
    if nodes_by_status:
        for status, cnt in nodes_by_status:
            print(f"  - {status}: {cnt}")
    else:
        print("  (no Act nodes found)")

    print("")
    print(f"Total edges (Act -> Act): {stats['total_edges']}")
    print(
        f"Relationship types present: {stats['distinct_relationship_types']}"
    )

    print("Edges by type:")
    edges_by_type: List[Tuple[str, int]] = stats["edges_by_type"]  # type: ignore[assignment]
    if edges_by_type:
        for rel_type, cnt in edges_by_type:
            print(f"  - {rel_type}: {cnt}")
    else:
        print("  (no edges found)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Neo4j graph stats: nodes, status breakdown, edges, and type counts."
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Optional Neo4j database name (overrides environment).",
    )
    parser.add_argument(
        "--acts-by-year-output",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "analytics", "acts_by_year.png")),
        help="Path to save the Acts-by-Year histogram PNG (default: outputs/analytics/acts_by_year.png)",
    )
    parser.add_argument(
        "--year-numbers",
        action="store_true",
        help="Print per-year counts to stdout (includes years with zero acts)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help=(
            "Apply moving-average smoothing with window N to line/area charts. "
            "Ignored for bar charts. Use odd N for centered smoothing."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        stats = compute_stats(database=args.database)
        print_stats(stats)

        # Render Acts-by-Year histogram
        out_path = args.acts_by_year_output
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with neo4j_session(database=args.database) as session:
            rows = fetch_acts_by_year(session)
            titles_by_year_rows = fetch_act_titles_by_year(session)
        if rows:
            years = [y for y, _ in rows]
            counts_map = {y: c for y, c in rows}
            min_year, max_year = min(years), max(years)
            all_years = list(range(min_year, max_year + 1))
            counts_raw = [counts_map.get(y, 0) for y in all_years]
            counts_smoothed = None
            if args.smooth and args.smooth > 1:
                counts_smoothed = _moving_average_centered([float(x) for x in counts_raw], int(args.smooth))

            # Optionally print per-year counts (including zero-filled years)
            if args.year_numbers:
                print("Acts by Year (inclusive range):")
                for y, c in zip(all_years, counts_raw):
                    print(f"  {y}: {c}")

            # Derive filenames
            base, ext = os.path.splitext(out_path)
            if not ext:
                ext = ".png"
            bar_path = out_path
            line_path = f"{base}_line{ext}"
            area_path = f"{base}_area{ext}"

            # Shared x-axis ticks logic
            def _apply_axis_common():
                ax = plt.gca()
                ax.set_xlim(min_year - 0.5, max_year + 0.5)
                step = 1
                if len(all_years) > 40:
                    step = 5
                if len(all_years) > 120:
                    step = 10
                xticks = [y for i, y in enumerate(all_years) if i % step == 0]
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(y) for y in xticks], rotation=45, ha="right", fontsize=9)
                ax.tick_params(axis='y', labelsize=10)

            # 1) Bar chart
            plt.figure(figsize=(14, 6), dpi=200)
            plt.bar(all_years, counts_raw, color="#4682B4", edgecolor="black", linewidth=0.4)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Number of Acts", fontsize=12)
            plt.title("New Zealand Legislative Acts by Year", fontsize=16, pad=10)
            _apply_axis_common()
            plt.tight_layout()
            plt.savefig(bar_path, format="PNG", bbox_inches="tight")
            plt.close()
            print(f"Saved Acts-by-Year bar chart -> {bar_path}")

            # 2) Line chart (optionally smoothed)
            plt.figure(figsize=(14, 6), dpi=200)
            series = counts_smoothed if counts_smoothed is not None else counts_raw
            marker = "o" if len(all_years) <= 80 else None
            plt.plot(
                all_years,
                series,
                color="#4682B4",
                linewidth=2.0,
                marker=marker,
                markersize=3,
            )
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Number of Acts", fontsize=12)
            plt.title("New Zealand Legislative Acts by Year", fontsize=16, pad=10)
            _apply_axis_common()
            plt.tight_layout()
            plt.savefig(line_path, format="PNG", bbox_inches="tight")
            plt.close()
            print(f"Saved Acts-by-Year line chart -> {line_path}")

            # 3) Area chart (optionally smoothed)
            plt.figure(figsize=(14, 6), dpi=200)
            series = counts_smoothed if counts_smoothed is not None else counts_raw
            plt.fill_between(all_years, series, step=None, alpha=0.35, color="#4682B4")
            plt.plot(all_years, series, color="#2f5f8a", linewidth=1.5)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Number of Acts", fontsize=12)
            plt.title("New Zealand Legislative Acts by Year", fontsize=16, pad=10)
            _apply_axis_common()
            plt.tight_layout()
            plt.savefig(area_path, format="PNG", bbox_inches="tight")
            plt.close()
            print(f"Saved Acts-by-Year area chart -> {area_path}")
        else:
            print("No Act year data found in Neo4j; skipped Acts-by-Year charts.")
        # Always export Excel (years as columns, rows are Act titles; skip year 0)
        excel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "analytics"))
        os.makedirs(excel_dir, exist_ok=True)
        excel_path = os.path.join(excel_dir, "acts_by_year.xlsx")
        grouped: Dict[int, List[str]] = {}
        for y, title in titles_by_year_rows:
            grouped.setdefault(y, []).append(title)
        years_sorted = sorted(grouped.keys())
        # Sort titles within each year and pad columns to equal length
        max_len = max((len(v) for v in grouped.values()), default=0)
        data: Dict[int, List[str]] = {}
        for y in years_sorted:
            titles = sorted(grouped[y])
            if len(titles) < max_len:
                titles = titles + [""] * (max_len - len(titles))
            data[y] = titles
        wide_df = pd.DataFrame(data)
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            wide_df.to_excel(writer, sheet_name="ByYear", index=False)
        print(f"Saved Acts-by-Year Excel (ByYear sheet) -> {excel_path}")
    finally:
        close_driver()


if __name__ == "__main__":
    main()
