from src.db.db_connection import db_connection


def ensure_lost_and_found_table():
    """Ensure the lost_and_found table exists (idempotent)."""
    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Cannot ensure lost_and_found table.")
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.lost_and_found');")
            exists = cursor.fetchone()[0]
            if not exists:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lost_and_found (
                        id SERIAL PRIMARY KEY,
                        object_name TEXT NOT NULL UNIQUE,
                        title_embedding REAL[],
                        found_title TEXT,
                        similarity REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE OR REPLACE FUNCTION set_updated_at_lost_and_found()
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
                    DROP TRIGGER IF EXISTS trigger_set_updated_at_lost_and_found ON lost_and_found;
                    """
                )
                cursor.execute(
                    """
                    CREATE TRIGGER trigger_set_updated_at_lost_and_found
                    BEFORE UPDATE ON lost_and_found
                    FOR EACH ROW
                    EXECUTE FUNCTION set_updated_at_lost_and_found();
                    """
                )
                print("lost_and_found table created.")

            conn.commit()

    except Exception as e:
        print(f"Error ensuring lost_and_found table: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_connection.release_connection(conn)
