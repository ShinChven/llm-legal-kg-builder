import argparse
import sys
from src.db.db_connection import db_connection

def get_act_full_text(title):
    """
    Retrieves the full text of an act from the database.
    """
    conn = db_connection.get_connection()
    if not conn:
        print("Database connection failed.")
        return None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT text FROM legislations WHERE title = %s", (title,))
            result = cursor.fetchone()
            return result[0] if result else None
    except Exception as e:
        print(f"Error fetching act text: {e}")
        return None
    finally:
        db_connection.release_connection(conn)

def get_act_relationships(title):
    """
    Retrieves all act relationships for a given act title.
    """
    conn = db_connection.get_connection()
    if not conn:
        print("Database connection failed.")
        return []
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT object_name FROM act_relationships WHERE subject_name = %s", (title,))
            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error fetching act relationships: {e}")
        return []
    finally:
        db_connection.release_connection(conn)

def delete_relationship(subject_name, object_name):
    """
    Deletes a specific relationship from the act_relationships table.
    """
    conn = db_connection.get_connection()
    if not conn:
        print("Database connection failed.")
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM act_relationships WHERE subject_name = %s AND object_name = %s", (subject_name, object_name))
            conn.commit()
            print(f"Deleted relationship: {subject_name} -> {object_name}")
    except Exception as e:
        print(f"Error deleting relationship: {e}")
        conn.rollback()
    finally:
        db_connection.release_connection(conn)

def main():
    parser = argparse.ArgumentParser(description="Remove hallucinatory act relationships from the database.")
    parser.add_argument("title", help="The title of the act to check.")
    args = parser.parse_args()

    title = args.title
    print(f"Checking for hallucinatory relationships for: {title}")

    full_text = get_act_full_text(title)
    if not full_text:
        print(f"Could not find full text for '{title}'. Exiting.")
        return

    relationships = get_act_relationships(title)
    if not relationships:
        print("No relationships found for this act.")
        return

    hallucinations = []
    for object_name in relationships:
        if object_name not in full_text:
            hallucinations.append(object_name)

    if not hallucinations:
        print("No hallucinatory relationships found.")
        return

    print("\nThe following related acts were not found in the full text and might be hallucinations:")
    for i, hall in enumerate(hallucinations):
        print(f"  {i+1}. {hall}")

    confirm = input("\nDo you want to delete these relationships from the database? (y/n): ").lower()
    if confirm == 'y':
        for hall in hallucinations:
            delete_relationship(title, hall)
        print("Deletion complete.")
    else:
        print("No changes were made.")

if __name__ == "__main__":
    main()
