from typing import Iterable, List, Tuple

from src.db.db_connection import db_connection
from src.db.create_act_topics_table import ensure_act_topics_table


class ActTopicHandler:
    """CRUD and helpers for act_topics table and related queries."""

    def __init__(self):
        self.db_conn = db_connection

    def get_connection(self):
        return self.db_conn.get_connection()

    def release_connection(self, conn):
        self.db_conn.release_connection(conn)

    def ensure_table(self):
        ensure_act_topics_table()

    def upsert_topics(self, act_title: str, topics: Iterable[Tuple[str, str, int]]):
        """
        Upsert topics for an act. `topics` is an iterable of (committee, topic, importance).
        On conflict, the importance is overwritten with the new value.
        """
        self.ensure_table()
        conn = self.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot upsert act topics.")
            return 0
        try:
            with conn.cursor() as cursor:
                rows: List[Tuple[str, str, str, int]] = [
                    (act_title, c, t, int(i)) for (c, t, i) in topics
                ]
                cursor.executemany(
                    """
                    INSERT INTO act_topics (act_title, committee, topic, importance)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (act_title, committee, topic)
                    DO UPDATE SET importance = EXCLUDED.importance,
                                  updated_at = CURRENT_TIMESTAMP
                    """,
                    rows,
                )
                conn.commit()
                print(f"Upserted {len(rows)} topic(s) for '{act_title}'.")
                return len(rows)
        except Exception as e:
            print(f"Error upserting topics for '{act_title}': {e}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if conn:
                self.release_connection(conn)

    def fetch_topics_for_act(self, act_title: str) -> List[Tuple[str, str, int]]:
        """Return (committee, topic, importance) rows for the given act title."""
        self.ensure_table()
        conn = self.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot fetch act topics.")
            return []
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT committee, topic, importance
                    FROM act_topics
                    WHERE act_title = %s
                    ORDER BY committee, topic
                    """,
                    (act_title,),
                )
                rows = cursor.fetchall()
                return [(r[0], r[1], int(r[2])) for r in rows]
        except Exception as e:
            print(f"Error fetching topics for '{act_title}': {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

    def delete_topics_for_act(self, act_title: str) -> int:
        """Delete all topic rows for the given act title. Returns number of rows removed."""
        self.ensure_table()
        conn = self.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot delete act topics.")
            return 0
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM act_topics
                    WHERE act_title = %s
                    """,
                    (act_title,),
                )
                deleted = cursor.rowcount or 0
                conn.commit()
                print(f"Deleted {deleted} topic(s) for '{act_title}'.")
                return deleted
        except Exception as e:
            print(f"Error deleting topics for '{act_title}': {e}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if conn:
                self.release_connection(conn)

    def fetch_all_act_titles_from_nz_source(self) -> List[str]:
        """Fetch distinct act titles from the NZ legislation website source in legislations table."""
        conn = self.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot fetch act titles.")
            return []
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT DISTINCT title
                    FROM legislations
                    WHERE source = 'legislation.govt.nz' AND text IS NOT NULL AND title IS NOT NULL
                    ORDER BY title
                    """
                )
                rows = cursor.fetchall()
                return [r[0] for r in rows]
        except Exception as e:
            print(f"Error fetching act titles: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

    def fetch_text_by_title(self, title: str) -> str:
        """Retrieve the full text for a given act title from legislations."""
        conn = self.get_connection()
        if conn is None:
            print("PostgreSQL connection not available. Cannot fetch act text.")
            return ""
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT text
                    FROM legislations
                    WHERE title = %s
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (title,),
                )
                row = cursor.fetchone()
                return row[0] if row and row[0] else ""
        except Exception as e:
            print(f"Error fetching text for '{title}': {e}")
            return ""
        finally:
            if conn:
                self.release_connection(conn)

