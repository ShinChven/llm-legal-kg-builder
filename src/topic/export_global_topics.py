"""
Export topics across the entire corpus (all Acts) to an Excel workbook.

This mirrors src.topic.export_leiden_community_topics but without community
grouping. It joins `act_topics` with `legislations` (for metadata like year)
to provide detailed rows and several useful aggregates for presentation.

Outputs
-------
- Excel workbook with sheets:
  - "topic_committee_overall": aggregated rows per (committee, topic)
      columns: committee, topic, acts_count, avg_importance, total_importance, unique_acts
  - "topics_overall": aggregated rows per topic across all committees
      columns: topic, acts_count, total_importance, avg_importance, committees, dominant_committee
  - "act_topics_detailed": detailed per-Act topic rows
      columns: act_title, year, status, committee, topic, importance
  - "top_topics_overall": single row of top N topics (topic_01..topic_N)
  - "top_committees_overall": single row of top N committees (committee_01..committee_N)
  - "topics_by_year" (optional): topic-year pivot for the top K topics by acts_count

Usage
-----
python -m src.topic.export_global_topics \
  [--min-importance 0] [--top-n 30] [--topk-year 20] \
  [--output outputs/analytics/global_topics_YYYYMMDD_HHMMSS.xlsx]

Environment
-----------
- Requires PostgreSQL connection env vars used by src.db.db_connection
  (POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD)
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd

from src.db.db_connection import db_connection
from src.db.create_act_topics_table import ensure_act_topics_table


def ensure_parent_dir(path: str) -> None:
    p = os.path.dirname(path)
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def default_output_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("outputs", "analytics", f"global_topics_{ts}.xlsx")


def fetch_detailed_rows(min_importance: int = 0) -> pd.DataFrame:
    """Fetch detailed act-topic rows joined with legislation metadata.

    Returns DataFrame with columns:
        act_title, year, status, committee, topic, importance
    """
    ensure_act_topics_table()

    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("PostgreSQL connection not available. Check .env settings.")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    at.act_title,
                    l.year,
                    COALESCE(l.stage, 'unknown') AS status,
                    at.committee,
                    at.topic,
                    at.importance
                FROM act_topics AS at
                LEFT JOIN legislations AS l
                  ON l.title = at.act_title
                WHERE at.importance >= %s
                ORDER BY at.act_title ASC, at.committee ASC, at.topic ASC
                """,
                (int(min_importance),),
            )
            rows: List[Tuple] = cur.fetchall() or []
            cols = ["act_title", "year", "status", "committee", "topic", "importance"]
            return pd.DataFrame(rows, columns=cols)
    finally:
        db_connection.release_connection(conn)


def aggregate_committee_topic(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per (committee, topic)."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "committee",
                "topic",
                "acts_count",
                "avg_importance",
                "total_importance",
                "unique_acts",
            ]
        )

    grouped = (
        df.groupby(["committee", "topic"]).agg(
            acts_count=("act_title", "nunique"),
            avg_importance=("importance", "mean"),
            total_importance=("importance", "sum"),
            unique_acts=("act_title", lambda s: ", ".join(sorted(set(s))[:15])),
        )
        .reset_index()
    )
    grouped["avg_importance"] = grouped["avg_importance"].round(2)
    grouped = grouped.sort_values(
        ["acts_count", "total_importance", "avg_importance", "committee", "topic"],
        ascending=[False, False, False, True, True],
    )
    return grouped


def aggregate_topic_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per topic across all committees, with dominant committee."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "topic",
                "acts_count",
                "total_importance",
                "avg_importance",
                "committees",
                "dominant_committee",
            ]
        )

    # Overall per-topic totals
    totals = (
        df.groupby("topic").agg(
            acts_count=("act_title", "nunique"),
            total_importance=("importance", "sum"),
            avg_importance=("importance", "mean"),
            committees=("committee", lambda s: ", ".join(sorted(set(map(str, s)))))
        )
        .reset_index()
    )

    # Dominant committee per topic based on acts_count under that committee
    ct = (
        df.groupby(["topic", "committee"]).agg(acts_count=("act_title", "nunique")).reset_index()
    )
    ct_sorted = ct.sort_values(["topic", "acts_count", "committee"], ascending=[True, False, True])
    dominant = ct_sorted.groupby("topic").first().reset_index()[["topic", "committee"]]
    dominant.columns = ["topic", "dominant_committee"]

    out = totals.merge(dominant, on="topic", how="left")
    out["avg_importance"] = out["avg_importance"].round(2)
    out = out.sort_values(
        ["acts_count", "total_importance", "avg_importance", "topic"],
        ascending=[False, False, False, True],
    )
    return out


def _one_row_top_labels(labels: List[str], prefix: str, top_n: Optional[int]) -> pd.DataFrame:
    if top_n and top_n > 0:
        labels = labels[: int(top_n)]
    cols = {f"{prefix}_{i:02d}": (labels[i - 1] if i - 1 < len(labels) else "") for i in range(1, (len(labels) if top_n is None else top_n) + 1)}
    if not cols:
        return pd.DataFrame(columns=[f"{prefix}_01"])  # keep shape stable
    return pd.DataFrame([{**cols}])


def topics_by_year(df: pd.DataFrame, topk_topics: int = 20) -> pd.DataFrame:
    """Return a topic vs year pivot for top K topics by acts_count. Requires year column."""
    if df.empty or "year" not in df.columns or df["year"].dropna().empty:
        return pd.DataFrame(columns=["topic"])  # empty placeholder

    # Determine top K topics by number of acts mentioning the topic
    topic_order = (
        df.groupby("topic")["act_title"].nunique().sort_values(ascending=False).head(int(topk_topics))
    )
    top_topics = set(topic_order.index.astype(str))
    sub = df[df["topic"].astype(str).isin(top_topics)].copy()
    sub = sub.dropna(subset=["year"])  # keep rows with year

    # Count acts per (topic, year)
    counts = (
        sub.groupby(["topic", "year"]).agg(acts_count=("act_title", "nunique")).reset_index()
    )
    pivot = counts.pivot(index="topic", columns="year", values="acts_count").fillna(0).astype(int)
    # Order rows by overall topic_order and columns by year ascending
    pivot = pivot.loc[[t for t in topic_order.index if t in pivot.index]]
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot.reset_index(inplace=True)
    return pivot


def run(min_importance: int = 0, output: Optional[str] = None, top_n: Optional[int] = 30, topk_year: int = 20) -> str:
    detailed = fetch_detailed_rows(min_importance=min_importance)
    if detailed.empty:
        raise RuntimeError("No act topic rows found. Ensure act_topics has data.")

    agg_ct = aggregate_committee_topic(detailed)
    agg_t = aggregate_topic_overall(detailed)

    # Build single-row top tables
    top_topics_labels = agg_t["topic"].astype(str).tolist()
    top_committees_labels = (
        detailed.groupby("committee")["act_title"].nunique().sort_values(ascending=False).index.astype(str).tolist()
    )
    top_topics_df = _one_row_top_labels(top_topics_labels, "topic", top_n)
    top_committees_df = _one_row_top_labels(top_committees_labels, "committee", top_n)

    # Topic-year pivot (optional, only if year exists)
    topics_year_df = topics_by_year(detailed, topk_topics=int(topk_year))

    out_path = output or default_output_path()
    ensure_parent_dir(out_path)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        agg_ct.to_excel(writer, sheet_name="topic_committee_overall", index=False)
        agg_t.to_excel(writer, sheet_name="topics_overall", index=False)
        detailed.to_excel(writer, sheet_name="act_topics_detailed", index=False)
        top_topics_df.to_excel(writer, sheet_name="top_topics_overall", index=False)
        top_committees_df.to_excel(writer, sheet_name="top_committees_overall", index=False)
        topics_year_df.to_excel(writer, sheet_name="topics_by_year", index=False)

    print(f"Wrote global topics to {out_path}")
    print("  - topic_committee_overall: per (committee, topic) aggregates")
    print("  - topics_overall: per-topic aggregates with dominant committee")
    print("  - act_topics_detailed: detailed act-topic rows")
    print("  - top_topics_overall / top_committees_overall: labels-only single row")
    print("  - topics_by_year: topic-year pivot for top K topics (if year present)")
    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export global topics (all Acts) to Excel")
    ap.add_argument("--min-importance", type=int, default=0, help="Keep topics with importance >= this value (default: 0)")
    ap.add_argument("--top-n", type=int, default=30, help="Number of top labels to include for topics/committees (default: 30)")
    ap.add_argument("--topk-year", type=int, default=20, help="Top K topics to include in the topic-year pivot (default: 20)")
    ap.add_argument("--output", type=str, default=None, help="Output Excel path (default: outputs/analytics/global_topics_YYYYMMDD_HHMMSS.xlsx)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(min_importance=int(args.min_importance), output=args.output, top_n=args.top_n, topk_year=int(args.topk_year))
    finally:
        db_connection.close_all_connections()


if __name__ == "__main__":
    main()

