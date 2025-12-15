import logging
from src.db.db_connection import db_connection

def similarity_fix():
    """
    Fixes the object_name in the act_relationships table based on the found_title
    from the lost_and_found table where the similarity is greater than 0.99.
    Handles unique constraint violations by deleting the old record if a corrected version already exists.
    """
    conn = None
    total_updated_rows = 0
    total_deleted_rows = 0
    try:
        conn = db_connection.get_connection()
        if conn is None:
            logging.error("Failed to get database connection.")
            return

        with conn.cursor() as cur:
            # Get the mapping of old names to new names, excluding no-op rows
            # where object_name equals found_title
            cur.execute(
                """
                SELECT object_name, found_title
                FROM lost_and_found
                WHERE similarity > 0.99
                  AND found_title IS NOT NULL
                  AND object_name <> found_title
                """
            )
            name_map = {row[0]: row[1] for row in cur.fetchall()}

            if not name_map:
                logging.info("No records to update.")
                return

            # Get all relationships that need updating
            cur.execute(
                "SELECT id, subject_name, object_name FROM act_relationships WHERE object_name IN %s",
                (tuple(name_map.keys()),)
            )
            relationships_to_fix = cur.fetchall()

            for rel_id, subject_name, old_object_name in relationships_to_fix:
                new_object_name = name_map[old_object_name]

                # Check if the corrected relationship already exists
                cur.execute(
                    "SELECT 1 FROM act_relationships WHERE subject_name = %s AND object_name = %s",
                    (subject_name, new_object_name)
                )
                exists = cur.fetchone()

                if exists:
                    # If it exists, delete the old one
                    logging.info(f"Deleting duplicate relationship: ('{subject_name}', '{old_object_name}')")
                    cur.execute(
                        "DELETE FROM act_relationships WHERE id = %s",
                        (rel_id,)
                    )
                    total_deleted_rows += cur.rowcount
                else:
                    # Otherwise, update it
                    logging.info(f"Updating '{old_object_name}' to '{new_object_name}' for subject '{subject_name}'")
                    cur.execute(
                        "UPDATE act_relationships SET object_name = %s WHERE id = %s",
                        (new_object_name, rel_id)
                    )
                    total_updated_rows += cur.rowcount

            conn.commit()
            logging.info(f"Similarity fix applied successfully.")
            logging.info(f"Total rows updated: {total_updated_rows}")
            logging.info(f"Total rows deleted: {total_deleted_rows}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_connection.release_connection(conn)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    similarity_fix()
