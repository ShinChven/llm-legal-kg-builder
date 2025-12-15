"""
Shared Neo4j connection utilities.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, TYPE_CHECKING

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

if TYPE_CHECKING:  # pragma: no cover - typing only
    from neo4j import Session

load_dotenv()


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DB_NAME") or os.getenv("NEO4J_DATABASE") or None

        missing = [
            name
            for name, value in (
                ("NEO4J_URI", uri),
                ("NEO4J_USER", user),
                ("NEO4J_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing required Neo4j environment variables: {', '.join(missing)}"
            )

        return cls(uri=uri, user=user, password=password, database=database)


_CONFIG: Optional[Neo4jConfig] = None
_DRIVER: Optional[Driver] = None


def get_config() -> Neo4jConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Neo4jConfig.from_env()
    return _CONFIG


def get_driver() -> Driver:
    global _DRIVER
    if _DRIVER is None:
        config = get_config()
        _DRIVER = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
    return _DRIVER


def close_driver() -> None:
    global _DRIVER
    if _DRIVER is not None:
        _DRIVER.close()
        _DRIVER = None


@contextmanager
def neo4j_session(*, database: Optional[str] = None, **kwargs) -> Iterator["Session"]:
    """
    Context manager yielding a Neo4j session.

    Additional keyword arguments are passed to `driver.session(...)`.
    """
    config = get_config()
    driver = get_driver()
    session = driver.session(database=database or config.database, **kwargs)
    try:
        yield session
    finally:
        session.close()


__all__ = [
    "Neo4jConfig",
    "get_config",
    "get_driver",
    "close_driver",
    "neo4j_session",
]
