
import sys
import termios
import tty
import pandas as pd
from src.db.db_connection import db_connection

def get_subject_acts_from_golden_standard(golden_standard_path: str = "golden-standard-test-dataset.xlsx"):
    """Reads all subject act titles from the golden standard Excel file."""
    try:
        xls = pd.ExcelFile(golden_standard_path)
    except FileNotFoundError:
        print(f"Golden standard file not found at: {golden_standard_path}")
        return None

    titles = []
    for sheet_name in xls.sheet_names:
        try:
            df_title = pd.read_excel(xls, sheet_name=sheet_name, usecols="A", header=None, nrows=1)
            if not df_title.empty and pd.notna(df_title.iloc[0, 0]):
                titles.append(str(df_title.iloc[0, 0]).strip())
            else:
                # Fallback to sheet name if A1 is empty
                titles.append(str(sheet_name))
        except Exception:
            titles.append(str(sheet_name))
    return titles

def interactive_menu(options, prompt="Select an item (use ↑/↓, Enter to choose):"):
    """A simple interactive menu for terminal selection."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        selected = 0
        # Initial render
        print(prompt)
        for i, opt in enumerate(options):
            if i == selected:
                print(f"> \x1b[7m{opt}\x1b[0m")
            else:
                print(f"  {opt}")

        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":  # Arrow key sequence
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':  # Up arrow
                        selected = (selected - 1) % len(options)
                    elif ch3 == 'B':  # Down arrow
                        selected = (selected + 1) % len(options)

                    # Redraw menu
                    print(f"\x1b[{len(options)}F", end='')
                    for i, opt in enumerate(options):
                        if i == selected:
                            print(f"> \x1b[7m{opt}\x1b[0m")
                        else:
                            print(f"  {opt}")

            elif ch in ('\r', '\n'):  # Enter key
                return options[selected]
            elif ch == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def delete_relationships_for_subject(subject_act_title: str):
    """Deletes all relationships for a given subject act from the database."""
    conn = db_connection.get_connection()
    if not conn:
        print("Database connection failed.")
        return False

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM act_relationships WHERE LOWER(subject_name) = LOWER(%s)",
                (subject_act_title,),
            )
            conn.commit()
            print(f"Deleted {cursor.rowcount} records for subject act: {subject_act_title}")
            return True
    except Exception as e:
        print(f"An error occurred while deleting relationships for {subject_act_title}: {e}")
        conn.rollback()
        return False
    finally:
        db_connection.release_connection(conn)

def main():
    """Main function to drive the cleaning process."""
    subject_acts = get_subject_acts_from_golden_standard()
    if not subject_acts:
        return

    menu_options = ["All", "Exit"] + sorted(subject_acts)

    try:
        print("--- Clean Evaluation Results ---")
        selection = interactive_menu(menu_options, "Select a subject act to clean results for:")

        if selection == "Exit":
            print("Exiting.")
            return

        print(f"You selected: {selection}")
        confirm = input("Are you sure you want to delete these records? This action cannot be undone. (y/n):")

        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return

        if selection == "All":
            print("Deleting records for all subject acts in the golden standard set...")
            for act_title in subject_acts:
                delete_relationships_for_subject(act_title)
            print("All specified records have been deleted.")
        else:
            delete_relationships_for_subject(selection)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
