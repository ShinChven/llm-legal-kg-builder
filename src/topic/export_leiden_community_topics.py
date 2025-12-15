"""
Export topics grouped by Leiden community to an Excel workbook.

This script joins the PostgreSQL tables `leiden_communities` and `act_topics`
to produce a report of topics observed within each community.

Outputs
-------
- Excel workbook with three sheets by default:
  - "topics_by_community": aggregated rows per (community, committee, topic)
      columns: community, community_size, committee, topic, acts_count,
               avg_importance, total_importance, unique_acts
  - "act_topics_by_community": detailed rows of each Act-topic assignment
      columns: community, community_size, act_title, status, committee, topic, importance
  - "top_topics_by_community": one row per community with its top N topics
      columns: community, community_size, topic_01 .. topic_N (ordered by topic popularity, topic labels only)
      If --top-n is omitted, exports all topics per community; N becomes the
      maximum topic count across communities in the sheet.
  - "top_committees_by_community": one row per community with its top N committees
      columns: community, community_size, committee_01 .. committee_N (ordered by committee popularity)

Usage
-----
python -m src.topic.export_leiden_community_topics \
  [--min-importance 0] [--top-n N] \
  [--output outputs/analytics/community_topics_latest.xlsx]
  If --top-n is not provided, exports all topics per community.

Environment
-----------
- Requires PostgreSQL connection env vars as used by src.db.db_connection
  (POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD)

Notes
-----
- Assumes `src/graph/leiden_community_detection.py` has been run and
  `leiden_communities` contains latest assignments.
- Assumes `act_topics` has been populated via the topic extraction pipeline.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd

from src.db.db_connection import db_connection
from src.db.create_leiden_communities_table import ensure_leiden_communities_table
from src.db.create_act_topics_table import ensure_act_topics_table


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def fetch_detailed_rows(min_importance: int = 0) -> pd.DataFrame:
    """Fetch detailed joined rows of Leiden community assignments and act topics.

    Returns DataFrame with columns:
        community, community_size, act_title, status, committee, topic, importance
    """
    # Ensure tables exist
    ensure_leiden_communities_table()
    ensure_act_topics_table()

    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("PostgreSQL connection not available. Check .env settings.")

    try:
        with conn.cursor() as cur:
            # Join leiden assignments with per‑act topics
            cur.execute(
                """
                SELECT
                    lc.community,
                    lc.community_size,
                    lc.act_title,
                    COALESCE(lc.status, 'unknown') AS status,
                    at.committee,
                    at.topic,
                    at.importance
                FROM leiden_communities AS lc
                JOIN act_topics AS at
                  ON at.act_title = lc.act_title
                WHERE at.importance >= %s
                ORDER BY lc.community ASC, lc.act_title ASC, at.committee ASC, at.topic ASC
                """,
                (int(min_importance),),
            )
            rows: List[Tuple] = cur.fetchall() or []
            cols = [
                "community",
                "community_size",
                "act_title",
                "status",
                "committee",
                "topic",
                "importance",
            ]
            df = pd.DataFrame(rows, columns=cols)
            return df
    finally:
        db_connection.release_connection(conn)


def aggregate_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate topics per (community, committee, topic).

    Produces columns:
        community, community_size, committee, topic,
        acts_count, avg_importance, total_importance, unique_acts
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "community",
                "community_size",
                "committee",
                "topic",
                "acts_count",
                "avg_importance",
                "total_importance",
                "unique_acts",
            ]
        )

    # Compute per (community, committee, topic) aggregation
    grouped = (
        df.groupby(["community", "committee", "topic"]).agg(
            acts_count=("act_title", "nunique"),
            avg_importance=("importance", "mean"),
            total_importance=("importance", "sum"),
            unique_acts=("act_title", lambda s: ", ".join(sorted(set(s))[:10])),
        )
        .reset_index()
    )

    # Attach community_size by joining any representative row per community
    comm_size = (
        df.groupby(["community"]).agg(community_size=("community_size", "max")).reset_index()
    )
    result = grouped.merge(comm_size, on="community", how="left")

    # Order columns and sort for readability
    result = result[
        [
            "community",
            "community_size",
            "committee",
            "topic",
            "acts_count",
            "avg_importance",
            "total_importance",
            "unique_acts",
        ]
    ]
    result = result.sort_values(["community", "committee", "acts_count"], ascending=[True, True, False])
    # Round averages for clean display
    result["avg_importance"] = result["avg_importance"].round(2)
    return result


def default_output_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("outputs", "analytics", f"community_topics_{ts}.xlsx")


def _top_topics_per_community(
    detailed: pd.DataFrame, top_n: int | None = None
) -> pd.DataFrame:
    """Return a DataFrame of one row per community with its top N topics.

    Ranking is by:
      1) acts_count (unique acts mentioning that committee-topic in the community)
      2) total_importance (sum across acts)
      3) avg_importance
    Cells contain "Committee: Topic" strings.
    Rows are ordered by community_size (desc).
    """
    if detailed.empty:
        # No topics to show; if top_n is None we don't know N in advance, so return only id/size columns
        base_cols = ["community", "community_size"]
        if top_n and top_n > 0:
            base_cols += [f"topic_{i:02d}" for i in range(1, top_n + 1)]
        return pd.DataFrame(columns=base_cols)

    # Aggregate at (community, committee, topic) granularity
    agg = (
        detailed.groupby(["community", "committee", "topic"]).agg(
            acts_count=("act_title", "nunique"),
            avg_importance=("importance", "mean"),
            total_importance=("importance", "sum"),
        )
    ).reset_index()

    # Join community_size
    comm_size = (
        detailed.groupby("community").agg(community_size=("community_size", "max")).reset_index()
    )
    agg = agg.merge(comm_size, on="community", how="left")

    # Build top-N list per community
    rows = []
    label_lists = []
    for comm_id, grp in agg.groupby("community"):
        grp_sorted = grp.sort_values(
            by=["acts_count", "total_importance", "avg_importance", "committee", "topic"],
            ascending=[False, False, False, True, True],
        )
        # Show topic labels only (no committee)
        labels = [f"{r.topic}" for r in grp_sorted.itertuples(index=False)]
        label_lists.append(labels)
        size_val = int(grp_sorted["community_size"].iloc[0]) if not grp_sorted.empty else None
        rows.append({"community": comm_id, "community_size": size_val, "__labels__": labels})

    # Determine final column width
    if top_n and top_n > 0:
        max_n = top_n
    else:
        max_n = max((len(ll) for ll in label_lists), default=0)

    # Materialize rows with padded/trimmed labels
    norm_rows = []
    for r in rows:
        labels = r.pop("__labels__")
        use_labels = (labels + [""] * max_n)[:max_n]
        for i, label in enumerate(use_labels, start=1):
            r[f"topic_{i:02d}"] = label
        norm_rows.append(r)

    out = pd.DataFrame(norm_rows)
    if not out.empty:
        out = out.sort_values(["community_size", "community"], ascending=[False, True])
        ordered_cols = ["community", "community_size"] + [f"topic_{i:02d}" for i in range(1, max_n + 1)]
        out = out[ordered_cols]
    return out


def _top_committees_per_community(
    detailed: pd.DataFrame, top_n: int | None = None
) -> pd.DataFrame:
    """Return a DataFrame of one row per community with its top N committees.

    Ranking is by acts_count (unique acts referencing any topic under that
    committee), then total_importance, then avg_importance. Cells contain
    committee names only. Rows ordered by community_size desc.
    """
    if detailed.empty:
        base_cols = ["community", "community_size"]
        if top_n and top_n > 0:
            base_cols += [f"committee_{i:02d}" for i in range(1, top_n + 1)]
        return pd.DataFrame(columns=base_cols)

    agg = (
        detailed.groupby(["community", "committee"]).agg(
            acts_count=("act_title", "nunique"),
            avg_importance=("importance", "mean"),
            total_importance=("importance", "sum"),
        )
    ).reset_index()

    comm_size = (
        detailed.groupby("community").agg(community_size=("community_size", "max")).reset_index()
    )
    agg = agg.merge(comm_size, on="community", how="left")

    rows = []
    label_lists = []
    for comm_id, grp in agg.groupby("community"):
        grp_sorted = grp.sort_values(
            by=["acts_count", "total_importance", "avg_importance", "committee"],
            ascending=[False, False, False, True],
        )
        labels = [f"{r.committee}" for r in grp_sorted.itertuples(index=False)]
        label_lists.append(labels)
        size_val = int(grp_sorted["community_size"].iloc[0]) if not grp_sorted.empty else None
        rows.append({"community": comm_id, "community_size": size_val, "__labels__": labels})

    if top_n and top_n > 0:
        max_n = top_n
    else:
        max_n = max((len(ll) for ll in label_lists), default=0)

    norm_rows = []
    for r in rows:
        labels = r.pop("__labels__")
        use_labels = (labels + [""] * max_n)[:max_n]
        for i, label in enumerate(use_labels, start=1):
            r[f"committee_{i:02d}"] = label
        norm_rows.append(r)

    out = pd.DataFrame(norm_rows)
    if not out.empty:
        out = out.sort_values(["community_size", "community"], ascending=[False, True])
        ordered_cols = ["community", "community_size"] + [f"committee_{i:02d}" for i in range(1, max_n + 1)]
        out = out[ordered_cols]
    return out


def _committee_dominance_table(detailed: pd.DataFrame) -> pd.DataFrame:
    """Return long-form committee dominance stats per community."""
    if detailed.empty:
        return pd.DataFrame(
            columns=[
                "community",
                "community_size",
                "committee",
                "acts_count",
                "pct_of_community",
                "rank",
            ]
        )

    grouped = (
        detailed.dropna(subset=["committee", "act_title"])
        .groupby(["community", "committee"])
        .agg(acts_count=("act_title", "nunique"))
        .reset_index()
    )

    comm_size = (
        detailed.groupby("community").agg(community_size=("community_size", "max")).reset_index()
    )
    result = grouped.merge(comm_size, on="community", how="left")
    result = result.dropna(subset=["committee"])

    # Compute dominance percentage and per-community rank
    result["pct_of_community"] = result.apply(
        lambda row: (row["acts_count"] / row["community_size"] * 100.0)
        if row.get("community_size") and row["community_size"] > 0
        else None,
        axis=1,
    )
    result = result.sort_values(
        ["community", "acts_count", "committee"], ascending=[True, False, True]
    )
    result["rank"] = (
        result.groupby("community")["acts_count"].rank(method="dense", ascending=False).astype(int)
    )
    ordered_cols = [
        "community",
        "community_size",
        "committee",
        "acts_count",
        "pct_of_community",
        "rank",
    ]
    return result[ordered_cols]


def run(min_importance: int = 0, output: str | None = None, top_n: int | None = None) -> str:
    detailed = fetch_detailed_rows(min_importance=min_importance)
    if detailed.empty:
        raise RuntimeError(
            "No joined rows found. Ensure leiden_communities and act_topics have data."
        )

    aggregated = aggregate_topics(detailed)
    top_topics = _top_topics_per_community(detailed, top_n=top_n)
    top_committees = _top_committees_per_community(detailed, top_n=top_n)
    committee_dominance = _committee_dominance_table(detailed)

    out_path = output or default_output_path()
    ensure_parent_dir(out_path)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        aggregated.to_excel(writer, sheet_name="topics_by_community", index=False)
        detailed.to_excel(writer, sheet_name="act_topics_by_community", index=False)
        top_topics.to_excel(writer, sheet_name="top_topics_by_community", index=False)
        top_committees.to_excel(writer, sheet_name="top_committees_by_community", index=False)
        committee_dominance.to_excel(writer, sheet_name="committee_dominance", index=False)

    print(f"Wrote community topics to {out_path}")
    print("  - Sheet 'topics_by_community': Aggregated topics per community")
    print("  - Sheet 'act_topics_by_community': Detailed act-topic rows per community")
    print("  - Sheet 'top_topics_by_community': Top topics per community (topic labels only)")
    print("  - Sheet 'top_committees_by_community': Top committees per community (one row each)")
    print("  - Sheet 'committee_dominance': Committee dominance metrics per community")
    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export topics of each Leiden community to Excel")
    ap.add_argument(
        "--min-importance",
        type=int,
        default=0,
        help="Filter to topics with importance >= this value (default: 0)",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=None,
        help=(
            "Number of top topics per community to export in one row. "
            "If omitted, exports all topics for each community."
        ),
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Excel filepath (default: outputs/analytics/community_topics_YYYYMMDD_HHMMSS.xlsx)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(min_importance=int(args.min_importance), output=args.output, top_n=args.top_n)
    finally:
        # Always close pooled connections on exit
        db_connection.close_all_connections()


if __name__ == "__main__":
    main()
