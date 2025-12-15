import argparse
import json
import os
import sys
import termios
import tty
from src.db.db_connection import db_connection
from src.evaluate.evaluation import verify_relationships
from rich.console import Console
from rich.table import Table

console = Console()

def get_relationships_from_db(subject_act_title: str):
    """
    Queries the database for all relationships of a given subject act.
    """
    conn = db_connection.get_connection()
    if not conn:
        print("Database connection failed.")
        return None

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT object_name, relationships
                FROM act_relationships
                WHERE LOWER(subject_name) = LOWER(%s)
                """,
                (subject_act_title,),
            )
            results = cursor.fetchall()

            if not results:
                return {}

            # The relationships are stored in a JSONB field.
            # We need to format it into Dict[str, List[str]]
            extracted_relationships = {}
            for row in results:
                object_name, relationships_json = row
                # Assuming relationships_json is a list of strings like '["R1", "R2"]'
                if isinstance(relationships_json, str):
                    relationships = json.loads(relationships_json)
                else: # It might already be a list/dict if psycopg2 handles JSONB well
                    relationships = relationships_json

                extracted_relationships[object_name] = relationships

            return extracted_relationships

    except Exception as e:
        print(f"An error occurred while fetching relationships: {e}")
        return None
    finally:
        db_connection.release_connection(conn)


def calculate_metrics(tp, fp, fn):
    """Calculates precision, recall, and F1-score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1_score


def main():
    parser = argparse.ArgumentParser(
        description="Verify extracted relationships for a given Act against the golden standard."
    )
    parser.add_argument(
        "act_title",
        type=str,
        nargs='?',
        default=None,
        help="The title of the subject Act to verify. If omitted, you'll be prompted to choose from the Excel sheets."
    )
    args = parser.parse_args()

    subject_act_title = args.act_title.strip() if args.act_title else None

    def read_a1_from_all_sheets(golden_standard_path: str):
        import pandas as pd

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
                    # fallback to sheet name if A1 empty
                    titles.append(str(sheet_name))
            except Exception:
                titles.append(str(sheet_name))

        return titles

    def interactive_menu(options, prompt="Select an item (use ↑/↓, Enter to choose):"):
        # Simple terminal arrow-key menu using termios/tty. First option should be Exit per requirement.
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
                if ch == "\x1b":
                    # possible arrow sequence
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'A':
                            selected = (selected - 1) % len(options)
                        elif ch3 == 'B':
                            selected = (selected + 1) % len(options)

                        # move cursor up block and redraw
                        print(f"\x1b[{len(options)}F", end='')
                        for i, opt in enumerate(options):
                            if i == selected:
                                print(f"> \x1b[7m{opt}\x1b[0m")
                            else:
                                print(f"  {opt}")

                elif ch in ('\r', '\n'):
                    return selected
                elif ch == '\x03':
                    raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # If no title provided, read A1 from all sheets and prompt user to choose
    if not subject_act_title:
        golden_path = "gold-standard-test-dataset.xlsx"
        titles = read_a1_from_all_sheets(golden_path)
        if titles is None:
            return

        options = ["Exit"] + titles
        try:
            choice = interactive_menu(options, prompt=f"Choose an Act from {golden_path} (Exit first):")
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            return

        if choice == 0:
            print("Exit selected. Exiting.")
            return

        subject_act_title = options[choice]
        print(f"Verifying relationships for: {subject_act_title}")
    # when provided on CLI the message was already printed above for interactive selection

    # 1. Get relationships from the database
    extracted_relationships = get_relationships_from_db(subject_act_title)

    if extracted_relationships is None:
        print("Could not retrieve relationships from the database. Exiting.")
        return

    if not extracted_relationships:
        print("No relationships found in the database for this Act.")
        # We can still run verification to see what's missing (false negatives)

    # 2. Verify against the golden standard
    verification_result = verify_relationships(
        subject_act_title=subject_act_title,
        extracted_relationships=extracted_relationships,
    )

    # 3. Print the results
    if "error" in verification_result:
        print(f"An error occurred during verification: {verification_result['error']}")
        return

    console.print("\n--- Verification Results ---", style="bold underline")
    # Pretty print the full verification results
    print(json.dumps(verification_result, indent=2))
    print("--------------------------\n")

    # --- Act Identification ---
    act_id_results = verification_result.get("act_identification", {})
    act_id_counts = act_id_results.get("counts", {})

    act_summary_table = Table(title="Act Identification Summary")
    act_summary_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    act_summary_table.add_column("Value", justify="left", style="magenta")
    act_summary_table.add_row(
        "Correctly Identified",
        f"{act_id_counts.get('correct', 0)} / {act_id_counts.get('golden', 0)}",
    )
    act_summary_table.add_row(
        "False Positives", str(act_id_counts.get("false_positives", 0))
    )
    act_summary_table.add_row(
        "False Negatives", str(act_id_counts.get("false_negatives", 0))
    )
    console.print(act_summary_table)

    # Calculate and print metrics for act identification
    act_tp = act_id_counts.get("correct", 0)
    act_fp = act_id_counts.get("false_positives", 0)
    act_fn = act_id_counts.get("false_negatives", 0)
    act_precision, act_recall, act_f1 = calculate_metrics(act_tp, act_fp, act_fn)

    act_metrics_table = Table(title="Act Identification Metrics")
    act_metrics_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    act_metrics_table.add_column("Score", justify="left", style="magenta")
    act_metrics_table.add_row("Precision", f"{act_precision:.2%}")
    act_metrics_table.add_row("Recall", f"{act_recall:.2%}")
    act_metrics_table.add_row("F1-score", f"{act_f1:.2%}")
    console.print(act_metrics_table)

    # --- Relationship Verification ---
    rel_counts = verification_result.get("counts", {})

    rel_summary_table = Table(title="Relationship Verification Summary")
    rel_summary_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    rel_summary_table.add_column("Value", justify="left", style="magenta")
    rel_summary_table.add_row(
        "Correctly Identified",
        f"{rel_counts.get('correct_relationships', 0)} / {rel_counts.get('golden_relationships', 0)}",
    )
    rel_summary_table.add_row(
        "False Positives", str(rel_counts.get("false_positives", 0))
    )
    rel_summary_table.add_row(
        "False Negatives", str(rel_counts.get("false_negatives", 0))
    )
    console.print(rel_summary_table)

    # Calculate and print metrics for relationship verification
    rel_tp = rel_counts.get("correct_relationships", 0)
    rel_fp = rel_counts.get("false_positives", 0)
    rel_fn = rel_counts.get("false_negatives", 0)
    rel_precision, rel_recall, rel_f1 = calculate_metrics(rel_tp, rel_fp, rel_fn)

    rel_metrics_table = Table(title="Relationship Verification Metrics")
    rel_metrics_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    rel_metrics_table.add_column("Score", justify="left", style="magenta")
    rel_metrics_table.add_row("Precision", f"{rel_precision:.2%}")
    rel_metrics_table.add_row("Recall", f"{rel_recall:.2%}")
    rel_metrics_table.add_row("F1-score", f"{rel_f1:.2%}")
    console.print(rel_metrics_table)

    # Optionally, print details of false positives and negatives if they exist
    if verification_result.get("false_positives"):
        console.print("\n[bold red]False Positive Relationships:[/bold red]")
        for item in verification_result["false_positives"]:
            console.print(f"- {item['object_act']}: {item['relationship']}")

    if verification_result.get("false_negatives"):
        console.print("\n[bold red]False Negative Relationships:[/bold red]")
        for item in verification_result["false_negatives"]:
            console.print(f"- {item['object_act']}: {item['relationship']}")



if __name__ == "__main__":
    main()
