"""
Construct a Neo4j graph from PostgreSQL, but only for maintained Acts
and relationships between maintained Acts.

This mirrors the general graph construction but filters to maintained items
based on the source hostname heuristic used in the main constructor.
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.graph.neo4j_connection import close_driver, neo4j_session

# Reuse helpers from the general constructor to avoid duplication
from src.graph.neo4j_graph_construction import (
    ActNode,
    RelationshipRecord,
    build_act_nodes,
    create_relationships,
    determine_status,
    ensure_constraints,
    fetch_all_legislation_metadata,
    fetch_relationship_records,
    upsert_act_nodes,
)


LOGGER = logging.getLogger(__name__)


def construct_graph_maintained(*, limit: Optional[int] = None) -> None:
    """
    Pull relationship data and metadata, filter to maintained-only, and push into Neo4j.
    - Nodes: only Acts whose chosen metadata resolves to status == 'maintained'.
    - Edges: only relationships where both subject and object are maintained Acts.
    """
    relationships_all = fetch_relationship_records(limit=limit)
    metadata_all = fetch_all_legislation_metadata()

    # Select only titles whose metadata resolves to maintained
    maintained_meta: Dict[str, Mapping[str, object]] = {
        title: meta for title, meta in metadata_all.items() if determine_status(meta.source) == "maintained"
    }
    maintained_titles: Set[str] = set(maintained_meta.keys())

    if not maintained_titles:
        LOGGER.info("No maintained acts found in legislations table; nothing to sync.")
        return

    # Keep only relationships fully within the maintained set
    relationships: List[RelationshipRecord] = [
        rec
        for rec in relationships_all
        if rec.subject in maintained_titles and rec.object in maintained_titles
    ]

    nodes: List[ActNode] = build_act_nodes(maintained_titles, maintained_meta)

    LOGGER.info(
        "Preparing to sync %d maintained act nodes and %d maintained relationship edges.",
        len(nodes),
        sum(len(record.codes) for record in relationships),
    )

    with neo4j_session() as session:
        ensure_constraints(session)
        upserted_nodes = upsert_act_nodes(session, nodes)
        LOGGER.info("Upserted %d maintained act nodes into Neo4j.", upserted_nodes)
        if relationships:
            merged_relationships = create_relationships(session, relationships)
            LOGGER.info("Merged %d maintained relationships into Neo4j.", merged_relationships)
        else:
            LOGGER.info("No maintained relationship rows to merge; only nodes were synced.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct a maintained-only Neo4j graph from PostgreSQL legislation data."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of relationship rows to process.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        construct_graph_maintained(limit=args.limit)
    except Exception as exc:
        LOGGER.exception("Failed to construct maintained-only Neo4j graph: %s", exc)
        raise
    finally:
        close_driver()


if __name__ == "__main__":
    main()

