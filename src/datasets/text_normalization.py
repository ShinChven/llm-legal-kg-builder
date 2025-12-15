"""
Text Normalization Script

Purpose:
    Some source datasets (e.g. NZ Government legislation site vs NZLII or other crawled XML/HTML feeds)
    may encode the apostrophe / single quote character differently:
        - RIGHT SINGLE QUOTATION MARK: U+2019  (’)
        - ASCII APOSTROPHE:            U+0027  (')

    When data is aggregated from multiple sources, these visually similar but distinct Unicode code points
    cause string mismatch issues for:
        * Title based JOIN / MATCH logic
        * Duplicate detection / deduping
        * Full‑text search token normalization
        * Downstream LLM prompt templating (mixed quoting can fragment tokens)

    This script normalizes curly apostrophes (’ U+2019) to the straight ASCII apostrophe (').

Operations performed (idempotent):
    1. UPDATE legislations.text  replacing all ’ with '
    2. UPDATE legislations.title replacing all ’ with '

    Each UPDATE only touches rows that actually contain the curly apostrophe (filtered with ILIKE '%’%')
    to avoid unnecessary writes / bloat in the table and WAL.

Usage:
    python -m src.datasets.text_normalization

Safety / Notes:
    - Run in a transaction; if any step fails the transaction is rolled back.
    - Prints per-column affected row counts.
    - Extend this module with additional normalization rules (e.g. smart quotes, non‑breaking spaces, em dashes)
      by appending to the NORMALIZATION_RULES structure.
"""

from dataclasses import dataclass
from typing import List
import psycopg2
from src.db.db_connection import db_connection


@dataclass(frozen=True)
class NormalizationRule:
    description: str
    column: str
    source: str  # substring / char to replace
    target: str
    where_pattern: str  # pattern for WHERE ... ILIKE '%…%'


# Future friendly: add more rules here (e.g., LEFT/RIGHT DOUBLE QUOTES, NBSP -> space, etc.)
NORMALIZATION_RULES: List[NormalizationRule] = [
    NormalizationRule(
        description="Normalize curly apostrophe to ASCII apostrophe in text",
        column="text",
        source="’",
        target="'",
        where_pattern="%’%",
    ),
    NormalizationRule(
        description="Normalize curly apostrophe to ASCII apostrophe in title",
        column="title",
        source="’",
        target="'",
        where_pattern="%’%",
    ),
]


def apply_rule(cursor, rule: NormalizationRule) -> int:
    """Execute a single normalization rule and return affected row count."""
    update_sql = f"""
        UPDATE legislations
        SET {rule.column} = REPLACE({rule.column}, %s, %s)
        WHERE {rule.column} ILIKE %s
    """
    cursor.execute(update_sql, (rule.source, rule.target, rule.where_pattern))
    return cursor.rowcount


def run_normalization(dry_run: bool = False) -> None:
    conn = db_connection.get_connection()
    if not conn:
        raise RuntimeError("Could not obtain a database connection from the pool.")

    try:
        cursor = conn.cursor()
        print("🧽 Starting text normalization (curly apostrophes -> ASCII apostrophes)...")

        total_changed = 0
        for rule in NORMALIZATION_RULES:
            print(f"\n➡️  Rule: {rule.description}")
            if dry_run:
                # preview how many rows would change
                preview_sql = f"SELECT COUNT(*) FROM legislations WHERE {rule.column} ILIKE %s"
                cursor.execute(preview_sql, (rule.where_pattern,))
                count = cursor.fetchone()[0]
                print(f"   (dry-run) Rows that would be updated: {count}")
            else:
                changed = apply_rule(cursor, rule)
                total_changed += changed
                print(f"   Updated rows: {changed}")

        if dry_run:
            print("\n🔎 Dry-run complete. No changes committed.")
            conn.rollback()
        else:
            conn.commit()
            print(f"\n✅ Normalization committed. Total rows updated across rules: {total_changed}")

    except Exception as e:
        print(f"❌ Error during normalization: {e}")
        conn.rollback()
        raise
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        db_connection.release_connection(conn)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Normalize text/title fields in legislations table.")
    parser.add_argument("--dry-run", action="store_true", help="Only show counts; don't apply updates")
    args = parser.parse_args()

    run_normalization(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
