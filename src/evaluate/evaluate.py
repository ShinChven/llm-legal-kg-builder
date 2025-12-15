#!/usr/bin/env python3
"""Evaluate all Acts in the gold-standard Excel and print a compact table of results.

This re-uses the verification logic from `src.evaluate.verification` and helpers in
`src.evaluate.verify` where possible. It prints one row per Act with NER and RE
precision/recall/F1.
"""
"""Module-level: evaluate all Acts using a hardcoded gold-standard path.
"""
from typing import List, Dict, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.re.act_relationship_handler import ActRelationshipHandler
from src.evaluate.evaluation import verify_relationships, evaluate_all as evaluate_all_total
from src.evaluate.verify import get_relationships_from_db, calculate_metrics

console = Console()


def read_a1_from_all_sheets(golden_standard_path: str) -> List[str]:
    """Read cell A1 from every sheet in the given Excel file to obtain Act titles.

    Falls back to the sheet name when A1 is empty or unreadable.
    """
    try:
        xls = pd.ExcelFile(golden_standard_path)
    except FileNotFoundError:
        console.print(f"Golden standard file not found at: {golden_standard_path}")
        return []

    titles: List[str] = []
    for sheet_name in xls.sheet_names:
        try:
            df_title = pd.read_excel(
                xls, sheet_name=sheet_name, usecols="A", header=None, nrows=1
            )
            if not df_title.empty and pd.notna(df_title.iloc[0, 0]):
                titles.append(str(df_title.iloc[0, 0]).strip())
            else:
                titles.append(str(sheet_name))
        except Exception:
            titles.append(str(sheet_name))

    return titles


def evaluate_and_print_summary(golden_path: str):
    titles = read_a1_from_all_sheets(golden_path)
    if not titles:
        console.print("No titles found in gold-standard file. Exiting.")
        return

    # Build golden counts (Acts, Relations) per core Act directly from the Excel dataset
    def golden_counts_by_title(path: str) -> Dict[str, Tuple[int, int]]:
        counts: Dict[str, Tuple[int, int]] = {}
        try:
            xls = pd.ExcelFile(path)
        except FileNotFoundError:
            console.print(f"Golden standard file not found at: {path}")
            return counts

        for sheet_name in xls.sheet_names:
            # Resolve subject/core Act title from A1, fallback to sheet name
            try:
                df_title = pd.read_excel(
                    xls, sheet_name=sheet_name, usecols="A", header=None, nrows=1
                )
                if not df_title.empty and pd.notna(df_title.iloc[0, 0]):
                    subject_title = str(df_title.iloc[0, 0]).strip()
                else:
                    subject_title = str(sheet_name)
            except Exception:
                subject_title = str(sheet_name)

            # Read full sheet to count golden Acts and Relations
            # Acts: number of UNIQUE normalized object Act titles (to align with evaluation logic)
            # Relations: sum of all '1' markers across relationship columns
            from src.re.normalization import normalize_for_comparison

            normalized_object_acts = set()
            unique_golden_rel_pairs = set()
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                # Iterate rows: first column is object act title, rest are relationship codes
                for _, row in df.iterrows():
                    object_act_title = row.iloc[0]
                    if pd.isna(object_act_title):
                        continue
                    norm_title = normalize_for_comparison(str(object_act_title).strip())
                    if norm_title:
                        normalized_object_acts.add(norm_title)
                    # Count cells marked as 1 / 1.0 across relationship columns
                    for idx, cell in enumerate(row.iloc[1:]):
                        if pd.notna(cell) and str(cell).strip() in ["1", "1.0"]:
                            # Relationship code from header position
                            code = str(df.columns[idx + 1]).strip()
                            unique_golden_rel_pairs.add((norm_title, code))
            except Exception:
                # On read error, leave counts as zero for this sheet
                pass

            counts[subject_title] = (len(normalized_object_acts), len(unique_golden_rel_pairs))

        return counts

    golden_counts = golden_counts_by_title(golden_path)

    # New density table
    density_table = Table(
        title="Density Summary",
    )
    density_table.add_column("Core Act", style="cyan")
    density_table.add_column("Word Count", justify="right")
    density_table.add_column("Acts", justify="right")
    density_table.add_column("Relations", justify="right")
    density_table.add_column("Acts / 1k words", justify="right")
    density_table.add_column("Relations / 1k words", justify="right")

    table = Table(
        title="Evaluation Summary",
    )
    table.add_column("Core Act", style="cyan")
    table.add_column("NER P", justify="right")
    table.add_column("NER R", justify="right")
    table.add_column("NER F1", justify="right")
    table.add_column("RE P", justify="right")
    table.add_column("RE R", justify="right")
    table.add_column("RE F1", justify="right")

    # New detailed summary table
    detailed_summary_table = Table(
        title="NER and RE Detailed Summary",
    )
    detailed_summary_table.add_column("Core Act", style="cyan")
    detailed_summary_table.add_column("NER Identified", justify="right")
    detailed_summary_table.add_column("NER FP", justify="right")
    detailed_summary_table.add_column("NER FN", justify="right")
    detailed_summary_table.add_column("RE Identified", justify="right")
    detailed_summary_table.add_column("RE FP", justify="right")
    detailed_summary_table.add_column("RE FN", justify="right")

    all_extracted: Dict[str, Dict] = {}
    total_word_count = 0
    total_object_acts = 0  # totals from golden dataset
    total_relationships = 0  # totals from golden dataset
    act_handler = ActRelationshipHandler()
    evaluation_rows = []
    density_rows = []
    detailed_rows = []

    for title in titles:
        # Fetch extracted relationships from DB
        extracted = get_relationships_from_db(title)
        all_extracted[title] = extracted if extracted is not None else {}

        word_count = act_handler.get_word_count_for_act(title) or 0
        total_word_count += word_count

        # Use golden dataset for density numbers
        golden_acts, golden_rels = golden_counts.get(title, (0, 0))
        total_object_acts += golden_acts
        total_relationships += golden_rels

        if extracted is None:
            # DB connection failure or similar
            act_p = act_r = act_f = rel_p = rel_r = rel_f = "ERR"
            detailed_row = [title, "ERR", "ERR", "ERR", "ERR", "ERR", "ERR"]
            # Density should reflect golden dataset even if DB fails
            if word_count > 0:
                density_row = [
                    title,
                    str(word_count),
                    str(golden_acts),
                    str(golden_rels),
                    f"{(golden_acts / word_count) * 1000:.2f}",
                    f"{(golden_rels / word_count) * 1000:.2f}",
                ]
            else:
                density_row = [title, str(word_count), str(golden_acts), str(golden_rels), "N/A", "N/A"]
        else:
            result = verify_relationships(
                subject_act_title=title, extracted_relationships=extracted
            )
            if not result or "error" in result:
                act_p = act_r = act_f = rel_p = rel_r = rel_f = "ERR"
                detailed_row = [title, "ERR", "ERR", "ERR", "ERR", "ERR", "ERR"]
                # Density still reflects golden dataset
                if word_count > 0:
                    density_row = [
                        title,
                        str(word_count),
                        str(golden_acts),
                        str(golden_rels),
                        f"{(golden_acts / word_count) * 1000:.2f}",
                        f"{(golden_rels / word_count) * 1000:.2f}",
                    ]
                else:
                    density_row = [title, str(word_count), str(golden_acts), str(golden_rels), "N/A", "N/A"]
            else:
                act_counts = result.get("act_identification", {}).get("counts", {})
                act_tp = act_counts.get("correct", 0)
                act_fp = act_counts.get("false_positives", 0)
                act_fn = act_counts.get("false_negatives", 0)
                a_p, a_r, a_f = calculate_metrics(act_tp, act_fp, act_fn)

                rel_counts = result.get("counts", {})
                rel_tp = rel_counts.get("correct_relationships", 0)
                rel_fp = rel_counts.get("false_positives", 0)
                rel_fn = rel_counts.get("false_negatives", 0)
                r_p, r_r, r_f = calculate_metrics(rel_tp, rel_fp, rel_fn)

                act_p = f"{a_p:.2%}"
                act_r = f"{a_r:.2%}"
                act_f = f"{a_f:.2%}"
                rel_p = f"{r_p:.2%}"
                rel_r = f"{r_r:.2%}"
                rel_f = f"{r_f:.2%}"

                ner_identified = (
                    f"{act_counts.get('correct', 0)} / {act_counts.get('golden', 0)}"
                )
                ner_fp = str(act_counts.get("false_positives", 0))
                ner_fn = str(act_counts.get("false_negatives", 0))
                # Use explicit golden relationship count as denominator for clarity
                golden_rel_total = rel_counts.get('golden_relationships', None)
                if golden_rel_total is None:
                    golden_rel_total = rel_counts.get('correct_relationships', 0) + rel_counts.get('false_negatives', 0)
                re_identified = (
                    f"{rel_counts.get('correct_relationships', 0)} / {golden_rel_total}"
                )
                re_fp = str(rel_counts.get("false_positives", 0))
                re_fn = str(rel_counts.get("false_negatives", 0))
                detailed_row = [
                    title,
                    ner_identified,
                    ner_fp,
                    ner_fn,
                    re_identified,
                    re_fp,
                    re_fn,
                ]

                # Density calculations using golden dataset (not identified/extracted)
                if word_count > 0:
                    obj_acts_per_1k = (golden_acts / word_count) * 1000
                    rels_per_1k = (golden_rels / word_count) * 1000
                    density_row = [
                        title,
                        str(word_count),
                        str(golden_acts),
                        str(golden_rels),
                        f"{obj_acts_per_1k:.2f}",
                        f"{rels_per_1k:.2f}",
                    ]
                else:
                    density_row = [
                        title,
                        str(word_count),
                        str(golden_acts),
                        str(golden_rels),
                        "N/A",
                        "N/A",
                    ]

        detailed_rows.append((word_count, detailed_row))
        density_rows.append((word_count, density_row))
        evaluation_rows.append(
            (word_count, [title, act_p, act_r, act_f, rel_p, rel_r, rel_f])
        )

    # Populate tables sorted by word count (descending)
    for _, row in sorted(density_rows, key=lambda item: item[0], reverse=True):
        density_table.add_row(*row)

    for _, row in sorted(evaluation_rows, key=lambda item: item[0], reverse=True):
        table.add_row(*row)

    for _, row in sorted(detailed_rows, key=lambda item: item[0], reverse=True):
        detailed_summary_table.add_row(*row)

    # Add total row for density table (golden-based numbers)
    if total_word_count > 0:
        total_obj_acts_per_1k = (total_object_acts / total_word_count) * 1000
        total_rels_per_1k = (total_relationships / total_word_count) * 1000
        density_table.add_section()
        density_table.add_row(
            "TOTAL",
            str(total_word_count),
            str(total_object_acts),
            str(total_relationships),
            f"{total_obj_acts_per_1k:.2f}",
            f"{total_rels_per_1k:.2f}",
            style="bold",
        )
    console.print(density_table)

    # Calculate and add total row
    merged_metrics_message = None
    total_results = evaluate_all_total(all_extracted, golden_path)
    if "error" not in total_results:
        act_counts = total_results.get("act_identification", {}).get("counts", {})
        rel_counts = total_results.get("relationship_identification", {}).get(
            "counts", {}
        )

        # Add total to detailed summary table
        detailed_summary_table.add_section()
        total_ner_identified = (
            f"{act_counts.get('correct', 0)} / {act_counts.get('golden', 0)}"
        )
        total_ner_fp = str(act_counts.get("false_positives", 0))
        total_ner_fn = str(act_counts.get("false_negatives", 0))
        total_re_identified = (
            f"{rel_counts.get('correct', 0)} / {rel_counts.get('golden', 0)}"
        )
        total_re_fp = str(rel_counts.get("false_positives", 0))
        total_re_fn = str(rel_counts.get("false_negatives", 0))
        detailed_summary_table.add_row(
            "TOTAL",
            total_ner_identified,
            total_ner_fp,
            total_ner_fn,
            total_re_identified,
            total_re_fp,
            total_re_fn,
            style="bold",
        )
        console.print(detailed_summary_table)

        act_tp = act_counts.get("correct", 0)
        act_fp = act_counts.get("false_positives", 0)
        act_fn = act_counts.get("false_negatives", 0)
        total_act_p, total_act_r, total_act_f = calculate_metrics(
            act_tp, act_fp, act_fn
        )

        rel_tp = rel_counts.get("correct", 0)
        rel_fp = rel_counts.get("false_positives", 0)
        rel_fn = rel_counts.get("false_negatives", 0)
        total_rel_p, total_rel_r, total_rel_f = calculate_metrics(
            rel_tp, rel_fp, rel_fn
        )

        table.add_section()
        table.add_row(
            "TOTAL",
            f"{total_act_p:.2%}",
            f"{total_act_r:.2%}",
            f"{total_act_f:.2%}",
            f"{total_rel_p:.2%}",
            f"{total_rel_r:.2%}",
            f"{total_rel_f:.2%}",
            style="bold",
        )
        # General micro-averaged end-to-end metric over triples (subject, object, relation)
        gen_precision, gen_recall, gen_f1 = calculate_metrics(
            rel_tp, rel_fp, rel_fn
        )
        merged_metrics_message = (
            "[bold]General (micro, triples)[/bold] "
            f"Precision: {gen_precision:.2%} | "
            f"Recall: {gen_recall:.2%} | "
            f"F1: {gen_f1:.2%}"
        )

    console.print(table)
    console.print(
        "[bold]Note:[/bold] The 'TOTAL' row in the Evaluation Summary is calculated based on the aggregated (atom) counts of True Positives, False Positives, and False Negatives across all Acts, not by averaging individual Act percentages."
    )
    if merged_metrics_message:
        console.print(merged_metrics_message)


def print_terminology_explanation():
    """Prints a simple explanation of the terminology used in the evaluation."""
    console.print(
        """
[bold]Explanation[/bold]

[bold]NER (Named Entity Recognition)[/bold]: Finding mentions of other Act titles in the text.
[bold]NER FP (False Positive)[/bold]: The program incorrectly flagged something as an Act title when it wasn't.
[bold]NER FN (False Negative)[/bold]: The program missed an Act title that was mentioned in the text.

[bold]RE (Relation Extraction)[/bold]: Figuring out how the Acts mentioned are related (e.g., one "amends" another).
[bold]RE FP (False Positive)[/bold]: The program reported a relationship between Acts that is wrong.
[bold]RE FN (False Negative)[/bold]: The program missed a relationship between Acts that was mentioned.

[bold]Precision[/bold]: Out of all the items the program found, how many were correct.
[bold]Recall[/bold]: Out of all the correct items that exist, how many did the program find.
[bold]F1-Score[/bold]: A single score that balances precision and recall for an overall measure of accuracy.
"""
    )


def main():
    # Hardcoded gold-standard testset path per user preference
    print_terminology_explanation()
    evaluate_and_print_summary("gold-standard-test-dataset.xlsx")


if __name__ == "__main__":
    main()
