from src.db.db_connection import db_connection


def _table_exists(cursor) -> bool:
    cursor.execute("SELECT to_regclass('public.re_layer_counts');")
    row = cursor.fetchone()
    # row should be a one-tuple like (None,) or ('re_layer_counts',)
    return bool(row and len(row) >= 1 and row[0])


def ensure_re_layer_counts_table():
    """Ensure the re_layer_counts table exists with the minimal count-only schema.

    Final schema (counts only):
      - core_act TEXT, layer SMALLINT, act_count INT, relationship_count INT, computed_at TIMESTAMPTZ
      - UNIQUE (core_act, layer)
    Includes best-effort migrations from earlier iterations.
    """
    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Cannot ensure re_layer_counts table.")
        return

    try:
        with conn.cursor() as cursor:
            exists = _table_exists(cursor)
            if not exists:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS re_layer_counts (
                        id BIGSERIAL PRIMARY KEY,
                        core_act TEXT NOT NULL,
                        layer SMALLINT NOT NULL,
                        act_count INTEGER NOT NULL,
                        relationship_count INTEGER NOT NULL,
                        computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (core_act, layer)
                    );
                    """
                )
                print("re_layer_counts table created.")
            else:
                # Migrate legacy columns to minimal names/types
                # 1) Rename acts_count -> act_count, relcode_count -> relationship_count
                cursor.execute(
                    """
                    DO $$ BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='re_layer_counts' AND column_name='acts_count'
                        ) AND NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='re_layer_counts' AND column_name='act_count'
                        ) THEN
                            EXECUTE 'ALTER TABLE re_layer_counts RENAME COLUMN acts_count TO act_count';
                        END IF;
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='re_layer_counts' AND column_name='relcode_count'
                        ) AND NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='re_layer_counts' AND column_name='relationship_count'
                        ) THEN
                            EXECUTE 'ALTER TABLE re_layer_counts RENAME COLUMN relcode_count TO relationship_count';
                        END IF;
                    END $$;
                    """
                )

                # 2) Drop layers_requested, total_acts, total_rel_codes if present
                for col in ("layers_requested", "total_acts", "total_rel_codes"):
                    cursor.execute(
                        """
                        DO $$ BEGIN
                            IF EXISTS (
                                SELECT 1 FROM information_schema.columns
                                WHERE table_name='re_layer_counts' AND column_name=%s
                            ) THEN
                                EXECUTE format('ALTER TABLE re_layer_counts DROP COLUMN %I', %s);
                            END IF;
                        END $$;
                        """,
                        (col, col),
                    )

                # 3) Ensure UNIQUE on (core_act, layer); drop legacy uniques then add if missing
                cursor.execute(
                    """
                    DO $$ DECLARE r RECORD; BEGIN
                        FOR r IN (
                            SELECT conname
                            FROM pg_constraint
                            WHERE conrelid = 're_layer_counts'::regclass
                              AND contype = 'u'
                              AND conname NOT LIKE 'ux_re_layer_counts_core_layer%'
                        ) LOOP
                            EXECUTE format('ALTER TABLE re_layer_counts DROP CONSTRAINT %I', r.conname);
                        END LOOP;
                    END $$;
                    """
                )
                # Add constraint only if it doesn't already exist
                cursor.execute(
                    """
                    DO $$ BEGIN
                        IF NOT EXISTS (
                            SELECT 1
                            FROM pg_constraint
                            WHERE conrelid = 're_layer_counts'::regclass
                              AND contype = 'u'
                              AND conname = 'ux_re_layer_counts_core_layer'
                        ) THEN
                            EXECUTE 'ALTER TABLE re_layer_counts ADD CONSTRAINT ux_re_layer_counts_core_layer UNIQUE (core_act, layer)';
                        END IF;
                    END $$;
                    """
                )

            # Helpful index
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_re_layer_counts_core_layer
                ON re_layer_counts (core_act, layer);
                """
            )
            conn.commit()
    except Exception as e:
        print(f"Error ensuring re_layer_counts table: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_connection.release_connection(conn)
