"""
Utilities for removing all data from a Neo4j database in batches.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

try:
    from .neo4j_connection import get_config, get_driver
except ImportError:  # pragma: no cover - allow running as a script
    # When executed directly (e.g. `python src/graph/neo4j_clean.py`)
    # the relative import is not available, so we add the folder to sys.path.
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.append(CURRENT_DIR)
    from neo4j_connection import get_config, get_driver

from neo4j.exceptions import Neo4jError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from neo4j import Session, Transaction

DEFAULT_BATCH_SIZE = 100


def clean_database(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    database: Optional[str] = None,
) -> int:
    """
    Delete all nodes (and relationships) from the selected Neo4j database.

    Returns the total number of nodes deleted.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    target_database = _resolve_database(database)
    driver = get_driver()
    total_deleted = 0

    try:
        with driver.session(database=target_database) as session:
            while True:
                deleted = _execute_delete_batch(session, batch_size=batch_size)
                if deleted == 0:
                    break
                total_deleted += deleted
    except Neo4jError as exc:  # pragma: no cover - runtime feedback
        if getattr(exc, "code", "") == "Neo.ClientError.Database.DatabaseNotFound":
            db_name = target_database or "(default database)"
            raise RuntimeError(
                f"Neo4j database '{db_name}' not found. "
                "Ensure NEO4J_DB_NAME/NEO4J_DATABASE matches an existing database "
                "or unset it to use the server default."
            ) from exc
        raise

    return total_deleted


def _execute_delete_batch(session: "Session", *, batch_size: int) -> int:
    executor = getattr(session, "execute_write", None)
    if callable(executor):
        return executor(_delete_batch, batch_size=batch_size)

    # Fallback for driver versions < 5
    return session.write_transaction(_delete_batch, batch_size=batch_size)


def _delete_batch(tx: "Transaction", *, batch_size: int) -> int:
    result = tx.run(
        """
        MATCH (n)
        WITH n LIMIT $batch_size
        WITH collect(n) AS batch
        FOREACH (node IN batch | DETACH DELETE node)
        RETURN size(batch) AS deleted_count
        """,
        batch_size=batch_size,
    )
    record = result.single()
    return int(record["deleted_count"]) if record and record["deleted_count"] else 0


def _resolve_database(target: Optional[str]) -> Optional[str]:
    if target:
        return target
    config = get_config()
    return config.database


__all__ = ["clean_database", "DEFAULT_BATCH_SIZE"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove all nodes and relationships from a Neo4j database."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of nodes to delete per batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--database",
        type=str,
        help="Explicit database name. If omitted, defaults to NEO4J_DB_NAME/NEO4J_DATABASE.",
    )

    args = parser.parse_args()
    deleted = clean_database(batch_size=args.batch_size, database=args.database)
    print(f"Deleted {deleted} nodes from Neo4j.")
