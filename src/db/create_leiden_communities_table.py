from src.db.db_connection import db_connection


def ensure_leiden_communities_table():
    """Ensure the leiden_communities table exists (idempotent)."""
    # Handle environments where DB pool is not initialised
    if not hasattr(db_connection, "get_connection"):
        print("PostgreSQL connection not available. Cannot ensure leiden_communities table.")
        return
    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Cannot ensure leiden_communities table.")
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.leiden_communities');")
            exists = cursor.fetchone()[0]
            if not exists:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS leiden_communities (
                        id SERIAL PRIMARY KEY,
                        act_title TEXT NOT NULL,
                        status TEXT,
                        community INTEGER NOT NULL,
                        community_size INTEGER NOT NULL,
                        intermediate_community_ids JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE OR REPLACE FUNCTION set_updated_at_leiden_communities()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                    """
                )
                cursor.execute(
                    """
                    DROP TRIGGER IF EXISTS trigger_set_updated_at_leiden_communities ON leiden_communities;
                    """
                )
                cursor.execute(
                    """
                    CREATE TRIGGER trigger_set_updated_at_leiden_communities
                    BEFORE UPDATE ON leiden_communities
                    FOR EACH ROW
                    EXECUTE FUNCTION set_updated_at_leiden_communities();
                    """
                )
                print("leiden_communities table created.")

            # Ensure helpful indexes (idempotent)
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_leiden_communities_act
                ON leiden_communities (act_title);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_leiden_communities_comm
                ON leiden_communities (community);
                """
            )
            conn.commit()

    except Exception as e:
        print(f"Error ensuring leiden_communities table: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_connection.release_connection(conn)
