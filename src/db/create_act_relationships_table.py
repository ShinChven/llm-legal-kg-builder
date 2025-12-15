from src.db.db_connection import db_connection


def ensure_act_relationships_table():
    """Ensure the act_relationships table exists (idempotent)."""
    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Cannot ensure act_relationships table.")
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.act_relationships');")
            exists = cursor.fetchone()[0]
            if not exists:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS act_relationships (
                        id SERIAL PRIMARY KEY,
                        subject_name TEXT NOT NULL,
                        object_name TEXT NOT NULL,
                        relationships JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(subject_name, object_name)
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE OR REPLACE FUNCTION set_updated_at_act_relationships()
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
                    DROP TRIGGER IF EXISTS trigger_set_updated_at_act_relationships ON act_relationships;
                    """
                )
                cursor.execute(
                    """
                    CREATE TRIGGER trigger_set_updated_at_act_relationships
                    BEFORE UPDATE ON act_relationships
                    FOR EACH ROW
                    EXECUTE FUNCTION set_updated_at_act_relationships();
                    """
                )
                print("act_relationships table created.")

            # Ensure helpful indexes (idempotent)
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_act_relationships_subject
                ON act_relationships (subject_name);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_act_relationships_object
                ON act_relationships (object_name);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_act_relationships_relationships_gin
                ON act_relationships USING GIN (relationships);
                """
            )
            conn.commit()

    except Exception as e:
        print(f"Error ensuring act_relationships table: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_connection.release_connection(conn)
