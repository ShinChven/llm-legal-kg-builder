import re
import sys
import os

from src.db.db_connection import db_connection

def clean_nzlii_titles(dry_run=True):
    """
    Cleans the web_title for documents from 'nzlii' source and updates the title field.
    It removes the last bracketed content from the web_title.
    """
    conn = None
    try:
        conn = db_connection.get_connection()
        cur = conn.cursor()

        # Fetch records from nzlii that might have scruffy titles
        cur.execute("SELECT id, web_title FROM legislation_documents WHERE title !~ '\\d{4}$' and source = 'nzlii'")
        records = cur.fetchall()

        update_count = 0
        for record in records:
            doc_id, web_title = record
            if web_title:
                # Remove the last bracketed content, e.g., (Local) (2 GEO V 1911 No 31)
                # The regex removes a space and then the last parentheses group
                cleaned_title = re.sub(r'\s*\([^)]*\)$', '', web_title).strip()

                # Further cleaning for titles that might have multiple bracketed legislative info
                # e.g. `... Act 1911 (Local) (2 GEO V 1911 No 31)`
                # This will remove the part ` (2 GEO V 1911 No 31)`
                cleaned_title = re.sub(r'\s*\([\d\sA-Z]+\sNo\s\d+\)$', '', cleaned_title).strip()


                # Check if the title has changed and update
                # Also check if the original title was different to avoid unnecessary updates
                # Using a select to get the current title to compare against
                cur.execute("SELECT title FROM legislation_documents WHERE id = %s", (doc_id,))
                current_title = cur.fetchone()[0]

                if cleaned_title != current_title:
                    if dry_run:
                        print(f"Dry run: Would update title for doc_id {doc_id}")
                        print(f"  Web title: {web_title}")
                        print(f"  Old title: {current_title}")
                        print(f"  New title: {cleaned_title}")
                    else:
                        cur.execute(
                            "UPDATE legislation_documents SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                            (cleaned_title, doc_id)
                        )
                    update_count += 1

        if not dry_run:
            conn.commit()
            print(f"Successfully cleaned and updated {update_count} titles.")
        else:
            print(f"\nDry run complete. {update_count} titles would be updated.")

    except Exception as e:
        print(f"An error occurred: {e}")
        if conn and not dry_run:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            db_connection.release_connection(conn)

if __name__ == '__main__':
    # Set dry_run to False to actually update the database
    clean_nzlii_titles(dry_run=False)
