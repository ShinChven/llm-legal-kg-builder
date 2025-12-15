import re
import sys
import os

from src.db.db_connection import db_connection

def clean_nzlii_title_types(dry_run=True):
    """
    Cleans the title field for documents by removing '(private)' or '(local)' from the title.
    Processes all records regardless of source.
    """
    conn = None
    try:
        conn = db_connection.get_connection()
        cur = conn.cursor()

        # First, let's check what records we have with (private) or (local) in titles
        print("=== Finding records with (private) or (local) in titles ===")

        # Query for (private) records
        cur.execute("SELECT id, title, web_title, source FROM legislation_documents WHERE title ILIKE '%(private)%'")
        private_records = cur.fetchall()
        print(f"Found {len(private_records)} records with '(private)' in title:")
        for record in private_records[:5]:  # Show first 5
            print(f"  ID: {record[0]}, Title: {record[1][:100]}..., Source: {record[3]}")
        if len(private_records) > 5:
            print(f"  ... and {len(private_records) - 5} more")

        # Query for (local) records
        cur.execute("SELECT id, title, web_title, source FROM legislation_documents WHERE title ILIKE '%(local)%'")
        local_records = cur.fetchall()
        print(f"\nFound {len(local_records)} records with '(local)' in title:")
        for record in local_records[:5]:  # Show first 5
            print(f"  ID: {record[0]}, Title: {record[1][:100]}..., Source: {record[3]}")
        if len(local_records) > 5:
            print(f"  ... and {len(local_records) - 5} more")

        # Now process all records that need cleaning
        print(f"\n=== Processing title cleaning (dry_run={dry_run}) ===")

        # Get all records that have (private) or (local) in their titles
        cur.execute("""
            SELECT id, title FROM legislation_documents
            WHERE title ILIKE '%(private)%' OR title ILIKE '%(local)%'
        """)
        records_to_clean = cur.fetchall()

        update_count = 0
        for record in records_to_clean:
            doc_id, title = record
            if title:
                # Remove (private) or (local) from the title (case insensitive)
                # This regex will match (private), (Private), (LOCAL), (local), etc.
                cleaned_title = re.sub(r'\s*\((private|local)\)\s*', ' ', title, flags=re.IGNORECASE).strip()

                # Clean up any double spaces that might result
                cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()

                # Check if the title has actually changed
                if cleaned_title != title:
                    if dry_run:
                        print(f"Dry run: Would update title for doc_id {doc_id}")
                        print(f"  Old title: {title}")
                        print(f"  New title: {cleaned_title}")
                        print()
                    else:
                        cur.execute(
                            "UPDATE legislation_documents SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                            (cleaned_title, doc_id)
                        )
                        print(f"Updated doc_id {doc_id}: {title} -> {cleaned_title}")
                    update_count += 1

        if not dry_run:
            conn.commit()
            print(f"\nSuccessfully cleaned and updated {update_count} titles.")
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

def show_sample_records():
    """
    Show sample records with (private) or (local) in titles for inspection.
    """
    conn = None
    try:
        conn = db_connection.get_connection()
        cur = conn.cursor()

        print("=== Sample records with (private) in title ===")
        cur.execute("SELECT id, title, web_title, source FROM legislation_documents WHERE title ILIKE '%(private)%' LIMIT 10")
        private_records = cur.fetchall()
        for record in private_records:
            print(f"ID: {record[0]}")
            print(f"Title: {record[1]}")
            print(f"Web Title: {record[2]}")
            print(f"Source: {record[3]}")
            print("-" * 80)

        print("\n=== Sample records with (local) in title ===")
        cur.execute("SELECT id, title, web_title, source FROM legislation_documents WHERE title ILIKE '%(local)%' LIMIT 10")
        local_records = cur.fetchall()
        for record in local_records:
            print(f"ID: {record[0]}")
            print(f"Title: {record[1]}")
            print(f"Web Title: {record[2]}")
            print(f"Source: {record[3]}")
            print("-" * 80)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            cur.close()
            db_connection.release_connection(conn)

if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--show-samples':
            show_sample_records()
        elif sys.argv[1] == '--dry-run':
            clean_nzlii_title_types(dry_run=True)
        elif sys.argv[1] == '--execute':
            print("WARNING: This will modify the database!")
            response = input("Are you sure you want to proceed? (yes/no): ")
            if response.lower() == 'yes':
                clean_nzlii_title_types(dry_run=False)
            else:
                print("Operation cancelled.")
        else:
            print("Usage:")
            print("  python src/datasets/clean_nzlii_title_types.py --show-samples  # Show sample records")
            print("  python src/datasets/clean_nzlii_title_types.py --dry-run       # Run in dry-run mode")
            print("  python src/datasets/clean_nzlii_title_types.py --execute       # Actually update the database")
    else:
        # Default to dry run
        print("Running in dry-run mode by default. Use --execute to actually update the database.")
        clean_nzlii_title_types(dry_run=True)
