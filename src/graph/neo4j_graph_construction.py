"""
Construct a Neo4j graph from the PostgreSQL legislation relationship data.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from src.db.create_act_relationships_table import ensure_act_relationships_table
from src.db.db_connection import db_connection
from src.graph.neo4j_connection import close_driver, neo4j_session

LOGGER = logging.getLogger(__name__)

MAINTAINED_HOSTNAMES = {"legislation.govt.nz", "www.legislation.govt.nz"}
YEAR_PATTERN = re.compile(r"(18|19|20)\d{2}(?!.*\d)")
DEFAULT_CHUNK_SIZE = 500


@dataclass(frozen=True)
class LegislationMetadata:
    title: str
    source: Optional[str]
    year: Optional[int]


@dataclass(frozen=True)
class ActNode:
    title: str
    year: Optional[int]
    status: str
    source: Optional[str] = None


@dataclass(frozen=True)
class RelationshipRecord:
    subject: str
    object: str
    codes: Tuple[str, ...]


def construct_graph(*, limit: Optional[int] = None) -> None:
    """
    Pull relationship data from PostgreSQL and push it into Neo4j.
    """
    relationships = fetch_relationship_records(limit=limit)
    all_metadata = fetch_all_legislation_metadata()

    relationship_titles: Set[str] = {
        record.subject for record in relationships
    } | {record.object for record in relationships}

    all_titles: Set[str] = set(all_metadata.keys()) | relationship_titles
    if not all_titles:
        LOGGER.info(
            "No acts found in legislations table or relationship records. Nothing to sync."
        )
        return

    missing_titles = relationship_titles - all_metadata.keys()
    if missing_titles:
        all_metadata.update(fetch_legislation_metadata(missing_titles))

    nodes = build_act_nodes(all_titles, all_metadata)

    LOGGER.info(
        "Preparing to sync %d act nodes and %d relationship edges.",
        len(nodes),
        sum(len(record.codes) for record in relationships),
    )

    with neo4j_session() as session:
        ensure_constraints(session)
        upserted_nodes = upsert_act_nodes(session, nodes)
        LOGGER.info("Upserted %d act nodes into Neo4j.", upserted_nodes)
        if relationships:
            merged_relationships = create_relationships(session, relationships)
            LOGGER.info("Merged %d relationships into Neo4j.", merged_relationships)
        else:
            LOGGER.info("No relationship rows to merge; only nodes were synced.")


def fetch_relationship_records(limit: Optional[int] = None) -> List[RelationshipRecord]:
    """
    Read subject/object/codes triples from the act_relationships table.
    """
    ensure_act_relationships_table()
    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("Failed to obtain PostgreSQL connection from pool.")

    query = """
        SELECT subject_name, object_name, relationships
        FROM act_relationships
        WHERE subject_name IS NOT NULL
          AND object_name IS NOT NULL
    """
    params: Sequence[object] = ()
    if limit is not None and limit > 0:
        query += " ORDER BY subject_name, object_name LIMIT %s"
        params = (limit,)

    records: List[RelationshipRecord] = []
    skipped_missing_codes = 0

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            for subject, obj, raw_codes in cursor.fetchall():
                subject_title = (subject or "").strip()
                object_title = (obj or "").strip()
                if not subject_title or not object_title:
                    continue

                decoded_codes = _coerce_relationship_codes(raw_codes)
                sanitized_codes = _sanitize_relationship_codes(decoded_codes)
                if not sanitized_codes:
                    skipped_missing_codes += 1
                    continue

                deduped_codes = tuple(dict.fromkeys(sanitized_codes))
                records.append(
                    RelationshipRecord(
                        subject=subject_title, object=object_title, codes=deduped_codes
                    )
                )
    finally:
        db_connection.release_connection(conn)

    if skipped_missing_codes:
        LOGGER.warning(
            "Skipped %d relationships with no usable relationship codes.",
            skipped_missing_codes,
        )

    LOGGER.info("Fetched %d relationship rows from PostgreSQL.", len(records))
    return records


def fetch_legislation_metadata(
    titles: Iterable[str],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, LegislationMetadata]:
    """
    Retrieve legislation metadata for the supplied titles.
    """
    title_list = sorted({title for title in titles if title})
    if not title_list:
        return {}

    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("Failed to obtain PostgreSQL connection for legislations.")

    metadata: Dict[str, LegislationMetadata] = {}
    hostnames_str = ", ".join([f"'{h}'" for h in MAINTAINED_HOSTNAMES])
    query = f"""
        SELECT DISTINCT ON (title) title, source, year
        FROM legislations
        WHERE title = ANY(%s)
        ORDER BY
            title,
            CASE
                WHEN LOWER(COALESCE(source, '')) IN ({hostnames_str}) THEN 1
                ELSE 0
            END DESC,
            year DESC NULLS LAST,
            updated_at DESC NULLS LAST
    """

    try:
        with conn.cursor() as cursor:
            for chunk in _chunked(title_list, chunk_size):
                cursor.execute(query, (chunk,))
                for title, source, year in cursor.fetchall():
                    metadata[title] = LegislationMetadata(
                        title=title, source=source, year=year
                    )
    finally:
        db_connection.release_connection(conn)

    LOGGER.info("Matched %d acts with metadata from legislations.", len(metadata))
    return metadata


def fetch_all_legislation_metadata() -> Dict[str, LegislationMetadata]:
    """
    Retrieve one metadata record for every distinct legislation title.
    """
    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("Failed to obtain PostgreSQL connection for legislations.")

    metadata: Dict[str, LegislationMetadata] = {}
    hostnames_str = ", ".join([f"'{h}'" for h in MAINTAINED_HOSTNAMES])
    query = f"""
        SELECT DISTINCT ON (title) title, source, year
        FROM legislations
        WHERE title IS NOT NULL AND btrim(title) <> ''
        ORDER BY
            title,
            CASE
                WHEN LOWER(COALESCE(source, '')) IN ({hostnames_str}) THEN 1
                ELSE 0
            END DESC,
            year DESC NULLS LAST,
            updated_at DESC NULLS LAST
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            for title, source, year in cursor.fetchall():
                metadata[title] = LegislationMetadata(
                    title=title, source=source, year=year
                )
    finally:
        db_connection.release_connection(conn)

    LOGGER.info("Fetched %d unique acts from legislations.", len(metadata))
    return metadata


def build_act_nodes(
    titles: Iterable[str],
    metadata: Mapping[str, LegislationMetadata],
) -> List[ActNode]:
    """
    Combine raw titles with metadata to create node payloads.
    """
    nodes: List[ActNode] = []
    for title in sorted({title for title in titles if title}):
        details = metadata.get(title)
        source = details.source if details else None
        year = details.year if details else None
        if year is None:
            year = extract_year_from_title(title)
        status = determine_status(source)
        nodes.append(ActNode(title=title, year=year, status=status, source=source))
    return nodes


def ensure_constraints(session) -> None:
    """
    Create helpful Neo4j constraints if they do not already exist.
    """
    session.run(
        """
        CREATE CONSTRAINT act_title_unique IF NOT EXISTS
        FOR (act:Act) REQUIRE act.title IS UNIQUE
        """
    )


def upsert_act_nodes(session, nodes: Sequence[ActNode]) -> int:
    """
    Merge act nodes into Neo4j.
    """
    if not nodes:
        return 0

    payload = [
        {
            "title": node.title,
            "year": node.year,
            "status": node.status,
            "source": node.source,
        }
        for node in nodes
    ]

    result = session.run(
        """
        UNWIND $nodes AS node
        MERGE (act:Act {title: node.title})
        SET
            act.year = node.year,
            act.status = node.status,
            act.source = node.source,
            act.last_synced_at = datetime()
        RETURN count(act) AS updated
        """,
        nodes=payload,
    )
    record = result.single()
    return int(record["updated"]) if record and record["updated"] is not None else 0


def create_relationships(
    session,
    relationships: Sequence[RelationshipRecord],
) -> int:
    """
    Merge relationship edges grouped by relationship code.
    """
    grouped: Dict[str, Set[Tuple[str, str]]] = {}
    for record in relationships:
        for code in record.codes:
            grouped.setdefault(code, set()).add((record.subject, record.object))

    total_pairs = 0
    for rel_type, pairs in grouped.items():
        rows = [{"subject": subj, "object": obj} for subj, obj in pairs]
        if not rows:
            continue
        _execute_write(session, _merge_relationships_of_type, rel_type, rows)
        total_pairs += len(rows)
    return total_pairs


def _merge_relationships_of_type(tx, rel_type: str, rows: Sequence[Mapping[str, str]]) -> None:
    """
    Merge a set of relationships for a specific relationship type.
    """
    tx.run(
        f"""
        UNWIND $rows AS row
        MATCH (subject:Act {{title: row.subject}})
        MATCH (object:Act {{title: row.object}})
        MERGE (subject)-[rel:{rel_type}]->(object)
        ON CREATE SET rel.created_at = datetime()
        SET rel.last_synced_at = datetime()
        """,
        rows=rows,
    )


def extract_year_from_title(title: str) -> Optional[int]:
    """
    Extract a trailing 4-digit year from an act title.
    """
    if not title:
        return None
    match = YEAR_PATTERN.search(title.strip())
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def determine_status(source: Optional[str]) -> str:
    """
    Determine act status based on the source hostname.
    """
    if not source:
        return "historical"
    normalized = source.strip().lower()
    normalized = re.sub(r"^https?://", "", normalized)
    host = normalized.split("/", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if host in {host_name.replace("www.", "") for host_name in MAINTAINED_HOSTNAMES}:
        return "maintained"
    return "historical"


def _coerce_relationship_codes(raw_value) -> List[str]:
    """
    Normalize the relationships column into a list of string codes.
    """
    if raw_value is None:
        return []

    if isinstance(raw_value, (list, tuple, set)):
        return [
            str(code).strip()
            for code in raw_value
            if code is not None and str(code).strip()
        ]

    if isinstance(raw_value, memoryview):
        raw_value = raw_value.tobytes()

    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = raw_value.decode("utf-8", errors="ignore")

    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return []
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [code.strip() for code in value.split(",") if code.strip()]
        if isinstance(decoded, list):
            return [
                str(code).strip()
                for code in decoded
                if code is not None and str(code).strip()
            ]
        return []

    return [str(raw_value).strip()]


def _sanitize_relationship_codes(codes: Sequence[str]) -> List[str]:
    """
    Sanitize raw relationship codes for use as Neo4j relationship types.
    """
    sanitized: List[str] = []
    for code in codes:
        cleaned = sanitize_relationship_type(code)
        if cleaned:
            sanitized.append(cleaned)
        else:
            LOGGER.debug("Discarded invalid relationship code: %r", code)
    return sanitized


def sanitize_relationship_type(code: str) -> Optional[str]:
    """
    Convert an arbitrary string code into a safe Neo4j relationship type.
    """
    if not code:
        return None
    candidate = re.sub(r"[^0-9A-Za-z]+", "_", str(code).upper())
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if not candidate:
        return None
    if candidate[0].isdigit():
        candidate = f"REL_{candidate}"
    return candidate


def _execute_write(session, func, *args, **kwargs):
    """
    Compatibility helper for execute_write/write_transaction across driver versions.
    """
    executor = getattr(session, "execute_write", None)
    if callable(executor):
        return executor(func, *args, **kwargs)
    return session.write_transaction(func, *args, **kwargs)


def _chunked(sequence: Sequence[str], size: int) -> Iterator[List[str]]:
    """
    Yield successive chunks from sequence.
    """
    for start in range(0, len(sequence), size):
        yield list(sequence[start : start + size])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct a Neo4j graph from PostgreSQL legislation data."
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
        construct_graph(limit=args.limit)
    except Exception as exc:
        LOGGER.exception("Failed to construct Neo4j graph: %s", exc)
        raise
    finally:
        close_driver()


if __name__ == "__main__":
    main()
