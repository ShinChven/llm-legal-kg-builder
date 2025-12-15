from src.db.db_connection import db_connection


def ensure_act_topics_table():
    """Ensure the act_topics table exists (idempotent) with useful indexes and triggers."""
    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Cannot ensure act_topics table.")
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.act_topics');")
            exists = cursor.fetchone()[0]
            if not exists:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS act_topics (
                        id SERIAL PRIMARY KEY,
                        act_title VARCHAR(512) NOT NULL,
                        committee VARCHAR(256) NOT NULL,
                        topic VARCHAR(256) NOT NULL,
                        importance INTEGER NOT NULL CHECK (importance BETWEEN 0 AND 100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (act_title, committee, topic)
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE OR REPLACE FUNCTION set_updated_at_act_topics()
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
                    DROP TRIGGER IF EXISTS trigger_set_updated_at_act_topics ON act_topics;
                    CREATE TRIGGER trigger_set_updated_at_act_topics
                    BEFORE UPDATE ON act_topics
                    FOR EACH ROW
                    EXECUTE FUNCTION set_updated_at_act_topics();
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_act_topics_act_title ON act_topics (act_title);
                    CREATE INDEX IF NOT EXISTS idx_act_topics_committee ON act_topics (committee);
                    CREATE INDEX IF NOT EXISTS idx_act_topics_topic ON act_topics (topic);
                    """
                )
                print("act_topics table created.")
        conn.commit()
    except Exception as e:
        print(f"Error ensuring act_topics table: {e}")
        if conn:
            conn.rollback()
    finally:
        db_connection.release_connection(conn)


if __name__ == "__main__":
    ensure_act_topics_table()

