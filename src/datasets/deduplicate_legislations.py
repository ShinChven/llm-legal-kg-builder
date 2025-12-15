"""
Legislation Deduplication Script

This script identifies and removes duplicate legislation entries from the 'legislations' table
based on the following (UPDATED) rules, now keeping the OLDEST available version instead of the newest:
1. Group by 'title' to find duplicates
2. For each group:
    - Prefer the lowest (oldest) numeric version (e.g. keep 1.0 over 2.0, 3.0, etc.)
    - If numeric versions exist, DELETE any 'latest' placeholder versions and higher numeric versions
    - If no numeric versions exist but 'latest' rows exist, keep the first 'latest' (or all if you prefer; current behaviour keeps all) and delete other non-numeric variants
    - If only non-numeric (and not 'latest') versions exist, keep the first row arbitrarily
3. Safely identifies rows to delete by their 'id' column
4. Provides option to execute deletions after review

Usage:
    python src/deduplicate_legislations.py
"""

import psycopg2
from src.db.db_connection import db_connection
from typing import List, Dict, Set
import re
from decimal import Decimal, InvalidOperation


def parse_version_number(version: str) -> Decimal:
    """
    Parse version string to Decimal for numeric comparison.

    Args:
        version: Version string (e.g., "1.0", "12.0", "16.5")

    Returns:
        Decimal representation of the version

    Raises:
        InvalidOperation: If version cannot be parsed as a number
    """
    if not version or version.lower() == 'latest':
        raise InvalidOperation("Cannot parse 'latest' or empty version as number")

    # Extract numeric part from version string
    # Handle cases like "1.0", "12", "16.5", etc.
    numeric_match = re.match(r'^(\d+(?:\.\d+)?)', str(version).strip())
    if not numeric_match:
        raise InvalidOperation(f"Cannot extract numeric value from version: {version}")

    return Decimal(numeric_match.group(1))


def load_legislations_data() -> List[Dict]:
    """
    Load all legislation data from the database.

    Returns:
        List of dictionaries containing id, title, and version for each row
    """
    conn = db_connection.get_connection()
    if not conn:
        raise Exception("Failed to get database connection")

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, version
            FROM legislations
            WHERE title IS NOT NULL
            ORDER BY title, version
        """)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        # Convert to list of dictionaries
        data = [dict(zip(columns, row)) for row in rows]

        print(f"Loaded {len(data)} legislation records from database")
        return data

    finally:
        cursor.close()
        db_connection.release_connection(conn)


def identify_duplicates_to_delete(data: List[Dict]) -> List[int]:
    """
    Identify duplicate legislation entries that should be deleted.

    Args:
        data: List of dictionaries with 'id', 'title', 'version' keys

    Returns:
        List of IDs to delete
    """
    # Group by title
    title_groups = {}
    for row in data:
        title = row['title']
        if title not in title_groups:
            title_groups[title] = []
        title_groups[title].append(row)

    ids_to_delete = []

    print(f"\nAnalyzing {len(title_groups)} unique titles...")

    for title, group in title_groups.items():
        if len(group) <= 1:
            # No duplicates, skip
            continue

        print(f"\n--- Processing title: '{title}' ({len(group)} versions) ---")

        # Separate components
        latest_rows = [row for row in group if row['version'] and row['version'].lower() == 'latest']
        numeric_rows = []  # list of tuples (row, version_decimal)
        non_numeric_rows: List[Dict] = []

        for row in group:
            version_val = row['version']
            if version_val and isinstance(version_val, str) and version_val.lower() == 'latest':
                # already captured in latest_rows
                continue
            try:
                version_num = parse_version_number(version_val)
                numeric_rows.append((row, version_num))
            except (InvalidOperation, TypeError):
                non_numeric_rows.append(row)
                print(f"    NOTE: Non-numeric version '{version_val}' for ID {row['id']}")

        if numeric_rows:
            # Find the LOWEST (oldest) numeric version to KEEP
            lowest_version = min(numeric_rows, key=lambda x: x[1])
            lowest_version_num = lowest_version[1]
            print(f"  Oldest numeric (kept) version: {lowest_version_num}")

            # Delete all numeric versions that are greater than the lowest
            for row, version_num in numeric_rows:
                if version_num > lowest_version_num:
                    ids_to_delete.append(row['id'])
                    print(f"    DELETE: ID {row['id']}, version '{row['version']}' (newer numeric: {version_num})")

            # Delete any 'latest' placeholder versions (they represent newer state)
            if latest_rows:
                print(f"  Found {len(latest_rows)} 'latest' placeholder version(s) -> deleting (keeping oldest numeric instead)")
                for row in latest_rows:
                    ids_to_delete.append(row['id'])
                    print(f"    DELETE: ID {row['id']}, version 'latest'")

            # Delete other non-numeric versions (ambiguous)
            for row in non_numeric_rows:
                ids_to_delete.append(row['id'])
                print(f"    DELETE: ID {row['id']}, version '{row['version']}' (non-numeric, not oldest numeric)")
        else:
            # No numeric versions exist
            if latest_rows and not non_numeric_rows:
                # Only 'latest' entries present – keep them all (or choose one). We'll keep all to avoid accidental loss.
                print(f"  Only 'latest' versions present ({len(latest_rows)} row(s)); keeping them (no numeric versions to compare).")
            elif latest_rows and non_numeric_rows:
                # Keep one non-latest? Simpler: keep first non-numeric (arbitrary oldest) and delete others including latest
                print(f"  No numeric versions; have {len(latest_rows)} 'latest' and {len(non_numeric_rows)} other non-numeric versions -> keeping first non-numeric, deleting others including all 'latest'.")
                # Keep first non-numeric
                keep_id = non_numeric_rows[0]['id'] if non_numeric_rows else latest_rows[0]['id']
                for row in non_numeric_rows:
                    if row['id'] != keep_id:
                        ids_to_delete.append(row['id'])
                        print(f"    DELETE: ID {row['id']}, version '{row['version']}' (extra non-numeric)")
                for row in latest_rows:
                    if row['id'] != keep_id:
                        ids_to_delete.append(row['id'])
                        print(f"    DELETE: ID {row['id']}, version 'latest' (preferring oldest arbitrary)")
            else:
                # All versions are non-numeric and none are 'latest' – keep first only
                print(f"  All versions are non-numeric (no 'latest', no numeric). Keeping first entry (ID {group[0]['id']})")
                for row in group[1:]:
                    ids_to_delete.append(row['id'])
                    print(f"    DELETE: ID {row['id']}, version '{row['version']}' (duplicate non-numeric)")

    return ids_to_delete


def preview_deletions(ids_to_delete: List[int]) -> None:
    """
    Show a preview of what will be deleted.

    Args:
        ids_to_delete: List of IDs that will be deleted
    """
    if not ids_to_delete:
        print("\n✅ No duplicates found. No deletions needed.")
        return

    print(f"\n📋 DELETION PREVIEW")
    print("=" * 50)
    print(f"Total rows to delete: {len(ids_to_delete)}")

    # Get details for preview
    conn = db_connection.get_connection()
    if not conn:
        print("❌ Failed to get database connection for preview")
        return

    try:
        cursor = conn.cursor()

        # Build query with placeholders
        placeholders = ','.join(['%s'] * len(ids_to_delete))
        cursor.execute(f"""
            SELECT id, title, version, created_at
            FROM legislations
            WHERE id IN ({placeholders})
            ORDER BY title, version
        """, ids_to_delete)

        rows = cursor.fetchall()

        print("\nRows to be deleted:")
        print("-" * 80)
        print(f"{'ID':<8} {'Title':<40} {'Version':<12} {'Created':<20}")
        print("-" * 80)

        for row in rows[:20]:  # Show first 20 for preview
            title = (row[1][:37] + '...') if len(row[1]) > 40 else row[1]
            print(f"{row[0]:<8} {title:<40} {row[2]:<12} {row[3]}")

        if len(rows) > 20:
            print(f"... and {len(rows) - 20} more rows")

    except Exception as e:
        print(f"❌ Error during preview: {e}")
    finally:
        cursor.close()
        db_connection.release_connection(conn)


def execute_deletions(ids_to_delete: List[int]) -> bool:
    """
    Execute the deletions in the database.

    Args:
        ids_to_delete: List of IDs to delete

    Returns:
        True if successful, False otherwise
    """
    if not ids_to_delete:
        print("No deletions to execute.")
        return True

    conn = db_connection.get_connection()
    if not conn:
        print("❌ Failed to get database connection for deletion")
        return False

    try:
        cursor = conn.cursor()

        # Build the DELETE query
        placeholders = ','.join(['%s'] * len(ids_to_delete))
        delete_query = f"DELETE FROM legislations WHERE id IN ({placeholders})"

        print(f"\n🗑️  Executing deletion of {len(ids_to_delete)} rows...")
        cursor.execute(delete_query, ids_to_delete)

        # Commit the transaction
        conn.commit()
        deleted_count = cursor.rowcount

        print(f"✅ Successfully deleted {deleted_count} rows")
        return True

    except Exception as e:
        print(f"❌ Error during deletion: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        db_connection.release_connection(conn)


def generate_sql_script(ids_to_delete: List[int], filename: str = "delete_duplicates.sql") -> None:
    """
    Generate a SQL script file for manual execution of deletions.

    Args:
        ids_to_delete: List of IDs to delete
        filename: Output filename for the SQL script
    """
    if not ids_to_delete:
        print("No deletions to generate SQL for.")
        return

    try:
        with open(filename, 'w') as f:
            f.write("-- Legislation Duplicate Deletion Script\n")
            f.write(f"-- Generated on: {__import__('datetime').datetime.now()}\n")
            f.write(f"-- Total rows to delete: {len(ids_to_delete)}\n\n")

            f.write("-- Preview what will be deleted:\n")
            ids_str = ','.join(map(str, ids_to_delete))
            f.write(f"SELECT id, title, version FROM legislations WHERE id IN ({ids_str});\n\n")

            f.write("-- Execute the deletion (uncomment to run):\n")
            f.write(f"-- DELETE FROM legislations WHERE id IN ({ids_str});\n\n")

            f.write("-- Verify deletion count:\n")
            f.write(f"-- Expected deleted rows: {len(ids_to_delete)}\n")

        print(f"📄 SQL script generated: {filename}")

    except Exception as e:
        print(f"❌ Error generating SQL script: {e}")


def main():
    """
    Main function to run the deduplication process.
    """
    print("🔍 NZ Legislation Deduplication Script")
    print("=" * 50)

    try:
        # Step 1: Load data
        print("\n1️⃣  Loading legislation data...")
        data = load_legislations_data()

        if not data:
            print("❌ No data found in legislations table")
            return

        # Step 2: Identify duplicates
        print("\n2️⃣  Identifying duplicates...")
        ids_to_delete = identify_duplicates_to_delete(data)

        # Step 3: Preview deletions
        print("\n3️⃣  Previewing deletions...")
        preview_deletions(ids_to_delete)

        if not ids_to_delete:
            return

        # Step 4: Generate SQL script
        print("\n4️⃣  Generating SQL script...")
        generate_sql_script(ids_to_delete)

        # Step 5: Ask user for confirmation
        print("\n5️⃣  Ready to execute deletions")
        print("⚠️  WARNING: This will permanently delete the identified duplicate rows!")

        while True:
            choice = input("\nProceed with deletion? (y/N): ").strip().lower()
            if choice in ['n', 'no', '']:
                print("❌ Deletion cancelled. SQL script is available for manual execution.")
                break
            elif choice in ['y', 'yes']:
                success = execute_deletions(ids_to_delete)
                if success:
                    print("✅ Deduplication completed successfully!")
                else:
                    print("❌ Deduplication failed. Check error messages above.")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n👋 Script completed.")


if __name__ == "__main__":
    main()
