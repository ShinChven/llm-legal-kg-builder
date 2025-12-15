import psycopg2
import psycopg2.extras
from src.db.db_connection import db_connection
import os
import datetime

def merge_relationships(relationships1, relationships2):
    """Merge two lists of relationships, avoiding duplicates."""
    merged = relationships1[:]
    for item in relationships2:
        if item not in merged:
            merged.append(item)
    return merged

def log_and_print(message, log_file):
    """Prints a message to the console and writes it to the log file."""
    print(message)
    if log_file:
        log_file.write(message + '\n')

def fix_titles_in_column(column_name, log_file):
    """Fix titles in a specific column (subject_name or object_name), with user confirmation."""
    conn = db_connection.get_connection()
    if not conn:
        log_and_print("Could not get a database connection.", log_file)
        return

    proposed_actions = []

    try:
        with conn.cursor() as cursor:
            # Find all names starting with "The "
            cursor.execute(f"SELECT DISTINCT {column_name} FROM act_relationships WHERE {column_name} LIKE 'The %%'")
            the_titles = [row[0] for row in cursor.fetchall()]

            for the_title in the_titles:
                plain_title = the_title[4:]

                # Determine the correct title from the 'legislations' table
                cursor.execute("SELECT title FROM legislations WHERE title = %s", (the_title,))
                is_the_title_correct = cursor.fetchone() is not None

                correct_title = the_title if is_the_title_correct else plain_title
                incorrect_title = plain_title if is_the_title_correct else the_title

                # Check if the plain title also exists to decide between merge or rename
                cursor.execute(f"SELECT 1 FROM act_relationships WHERE {column_name} = %s", (plain_title,))
                plain_title_exists = cursor.fetchone() is not None

                if plain_title_exists:
                    # MERGE PATH
                    log_and_print(f"Found pair: '{the_title}' and '{plain_title}'. Correct title: '{correct_title}'", log_file)

                    cursor.execute(f"SELECT id, subject_name, object_name, relationships FROM act_relationships WHERE {column_name} IN (%s, %s)", (the_title, plain_title))
                    records_to_process = cursor.fetchall()

                    grouped_records = {}
                    for record_id, subject, object_name, relationships in records_to_process:
                        other_col_val = subject if column_name == 'object_name' else object_name
                        if other_col_val not in grouped_records:
                            grouped_records[other_col_val] = {'correct': None, 'incorrect': None}

                        current_title_val = subject if column_name == 'subject_name' else object_name
                        if current_title_val == correct_title:
                            grouped_records[other_col_val]['correct'] = (record_id, relationships)
                        else:
                            grouped_records[other_col_val]['incorrect'] = (record_id, relationships)

                    for other_col_val, titles in grouped_records.items():
                        if titles['correct'] and titles['incorrect']:
                            _, correct_rels = titles['correct']
                            incorrect_id, incorrect_rels = titles['incorrect']
                            merged_rels = merge_relationships(correct_rels, incorrect_rels)
                            update_query_other_col = 'subject_name' if column_name == 'object_name' else 'object_name'

                            proposed_actions.append({
                                'type': 'merge',
                                'description': f"Merge relationships for '{other_col_val}' from '{incorrect_title}' into '{correct_title}'.",
                                'update_sql': f"UPDATE act_relationships SET relationships = %s WHERE {column_name} = %s AND {update_query_other_col} = %s",
                                'update_params': (psycopg2.extras.Json(merged_rels), correct_title, other_col_val),
                                'delete_sql': "DELETE FROM act_relationships WHERE id = %s",
                                'delete_params': (incorrect_id,)
                            })

                        elif titles['incorrect'] and not titles['correct']:
                            incorrect_id, _ = titles['incorrect']
                            update_query_other_col = 'subject_name' if column_name == 'object_name' else 'object_name'
                            cursor.execute(f"SELECT id FROM act_relationships WHERE {column_name} = %s AND {update_query_other_col} = %s", (correct_title, other_col_val))
                            if not cursor.fetchone():
                                proposed_actions.append({
                                    'type': 'rename',
                                    'description': f"Rename '{incorrect_title}' to '{correct_title}' for relationship with '{other_col_val}'.",
                                    'update_sql': f"UPDATE act_relationships SET {column_name} = %s WHERE id = %s",
                                    'update_params': (correct_title, incorrect_id)
                                })
                            else:
                                log_and_print(f"  Conflict: Target for rename '{correct_title}' already exists for '{other_col_val}'. Skipping rename for record ID {incorrect_id}.", log_file)

                elif correct_title != the_title:
                    # RENAME PATH
                    log_and_print(f"Found standalone incorrect title: '{the_title}'. Renaming to '{correct_title}'.", log_file)
                    cursor.execute(f"SELECT { 'subject_name' if column_name == 'object_name' else 'object_name' } FROM act_relationships WHERE {column_name} = %s", (correct_title,))
                    existing_partners = {row[0] for row in cursor.fetchall()}
                    cursor.execute(f"SELECT id, { 'subject_name' if column_name == 'object_name' else 'object_name' } FROM act_relationships WHERE {column_name} = %s", (the_title,))
                    records_to_rename = cursor.fetchall()

                    for record_id, partner_val in records_to_rename:
                        if partner_val in existing_partners:
                            log_and_print(f"  Conflict: Cannot rename for record ID {record_id} because a relationship between '{correct_title}' and '{partner_val}' already exists.", log_file)
                        else:
                            proposed_actions.append({
                                'type': 'rename',
                                'description': f"Rename standalone '{the_title}' to '{correct_title}' for relationship with '{partner_val}'.",
                                'update_sql': f"UPDATE act_relationships SET {column_name} = %s WHERE id = %s",
                                'update_params': (correct_title, record_id)
                            })

            if not proposed_actions:
                log_and_print(f"No changes needed for column '{column_name}'.", log_file)
                return

            log_and_print("\n--- Proposed Changes ---", log_file)
            for i, action in enumerate(proposed_actions, 1):
                log_and_print(f"{i}. {action['description']}", log_file)

            user_input = input("\nProceed with these changes? (y/n): ").lower()
            if user_input not in ['y', 'yes']:
                log_and_print("\nUser aborted. No changes were made.", log_file)
                # We need to rollback any potential transaction state if any read operation started one
                conn.rollback()
                return

            log_and_print("\nUser confirmed. Applying changes...", log_file)
            for action in proposed_actions:
                cursor.execute(action['update_sql'], action['update_params'])
                if action['type'] == 'merge':
                    cursor.execute(action['delete_sql'], action['delete_params'])

            conn.commit()
            log_and_print(f"Finished processing for column '{column_name}'.", log_file)

    except (Exception, psycopg2.DatabaseError) as error:
        log_and_print(f"Error while fixing titles in {column_name}: {error}", log_file)
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_connection.release_connection(conn)

def main():
    output_dir = './outputs/re/fix/'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dir, f"fix_log_{timestamp}.txt")

    with open(log_file_path, 'w') as log_file:
        log_and_print(f"Logging corrections to {log_file_path}", log_file)

        log_and_print("\nStarting title fix for 'object_name'...", log_file)
        fix_titles_in_column('object_name', log_file)

        log_and_print("\nStarting title fix for 'subject_name'...", log_file)
        fix_titles_in_column('subject_name', log_file)

    print(f"\nProcess complete. Log saved to {log_file_path}")
    db_connection.close_all_connections()

if __name__ == "__main__":
    main()
