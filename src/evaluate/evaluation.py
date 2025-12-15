import pandas as pd
from typing import Dict, List, Any, Tuple

from src.re.normalization import normalize_for_comparison


def verify_relationships(
    subject_act_title: str,
    extracted_relationships: Dict[str, List[str]],
    golden_standard_path: str = "gold-standard-test-dataset.xlsx"
) -> Dict[str, Any]:
    # The hardcoded xlsx file is the test dataset.
    # There are sheets in the xlsx, each is a manually labelled test set for a core act.
    # A1 cell is the title of the core Act due to sheet name is not allowed to input longer text.
    # We use the string in the cell to match evaluation target.
    """
    Verifies extracted relationships for a subject Act against a golden standard dataset in an Excel file.

    Args:
        subject_act_title (str): The title of the subject Act.
        extracted_relationships (Dict[str, List[str]]): A dictionary of extracted relationships.
            Keys are object Act titles (str), values are lists of relationship codes (str).
        golden_standard_path (str): The path to the golden standard Excel file.

    Returns:
        Dict[str, Any]: A dictionary containing the verification results, including
                         correctly identified relationships, false positives, and false negatives.
                         Returns an error message if the sheet for the subject Act is not found.
    """
    try:
        xls = pd.ExcelFile(golden_standard_path)
    except FileNotFoundError:
        return {"error": f"Golden standard file not found at: {golden_standard_path}"}

    target_sheet_name = None
    normalized_subject = normalize_for_comparison(subject_act_title)
    for sheet_name in xls.sheet_names:
        try:
            # Read only the first cell A1 to check the title
            df_title = pd.read_excel(xls, sheet_name=sheet_name, usecols="A", header=None, nrows=1)
            if df_title.empty:
                continue
            cell_value = df_title.iloc[0, 0]
            if pd.isna(cell_value):
                continue
            if normalize_for_comparison(str(cell_value)) == normalized_subject:
                target_sheet_name = sheet_name
                break
        except Exception:
            # Ignore sheets that can't be read or are empty
            continue

    if not target_sheet_name:
        return {"error": f"Sheet for subject Act '{subject_act_title}' not found in {golden_standard_path}."}

    # Load the full sheet now that we've found it
    df = pd.read_excel(xls, sheet_name=target_sheet_name)

    # Get relationship codes from the header (B1, C1, D1, E1, etc.)
    relationship_codes = [str(code) for code in df.columns[1:]]

    golden_data: Dict[str, List[str]] = {}
    original_case_golden: Dict[str, str] = {}
    for _, row in df.iterrows():
        object_act_title = row.iloc[0]
        if pd.isna(object_act_title):
            continue

        original_title = str(object_act_title).strip()
        normalized_title = normalize_for_comparison(original_title)
        if not normalized_title:
            continue

        if normalized_title not in golden_data:
            golden_data[normalized_title] = []
            original_case_golden[normalized_title] = original_title

        for i, code in enumerate(relationship_codes):
            cell_value = row.iloc[i + 1]
            # Check for 1, '1', 1.0, etc.
            if pd.notna(cell_value) and str(cell_value).strip() in ['1', '1.0']:
                golden_data[normalized_title].append(str(code).strip())

    # Normalize extracted relationships for comparison (keys to lowercase)
    normalized_extracted_relationships: Dict[str, List[str]] = {}
    original_case_extracted: Dict[str, str] = {}
    for key, values in extracted_relationships.items():
        normalized_key = normalize_for_comparison(str(key))
        if not normalized_key:
            continue
        normalized_values = [str(v).strip() for v in values]
        if normalized_key not in normalized_extracted_relationships:
            normalized_extracted_relationships[normalized_key] = []
            original_case_extracted[normalized_key] = str(key).strip()
        normalized_extracted_relationships[normalized_key].extend(normalized_values)

    # --- Act Name Identification ---
    extracted_act_titles = set(normalized_extracted_relationships.keys())
    golden_act_titles = set(golden_data.keys())

    correctly_identified_acts = sorted(list(extracted_act_titles.intersection(golden_act_titles)))
    false_positive_acts = sorted(list(extracted_act_titles.difference(golden_act_titles)))
    false_negative_acts = sorted(list(golden_act_titles.difference(extracted_act_titles)))

    act_identification_results = {
        "correctly_identified_acts": correctly_identified_acts,
        "false_positive_acts": false_positive_acts,
        "false_negative_acts": false_negative_acts,
        "counts": {
            "correct": len(correctly_identified_acts),
            "false_positives": len(false_positive_acts),
            "false_negatives": len(false_negative_acts),
            "extracted": len(extracted_act_titles),
            "golden": len(golden_act_titles),
        }
    }
    # --- End Act Name Identification ---

    # Build sets of relationships for deduplicated evaluation
    all_extracted_relations: List[Tuple[str, str]] = []
    for obj_act, rels in normalized_extracted_relationships.items():
        for rel in rels:
            all_extracted_relations.append((obj_act, rel))
    all_extracted_relations_set = set(all_extracted_relations)

    all_golden_relations: List[Tuple[str, str]] = []
    for obj_act, rels in golden_data.items():
        for rel in rels:
            all_golden_relations.append((obj_act, rel))
    all_golden_relations_set = set(all_golden_relations)

    correct_relationships_set = all_extracted_relations_set.intersection(all_golden_relations_set)
    false_positives_set = all_extracted_relations_set.difference(all_golden_relations_set)
    false_negatives_set = all_golden_relations_set.difference(all_extracted_relations_set)

    # Original case data for reporting
    def restore_case(item):
        lower_obj_act = item["object_act"]
        # Restore from extracted first, then golden
        original_obj_act = original_case_extracted.get(lower_obj_act, original_case_golden.get(lower_obj_act, lower_obj_act))
        return {"object_act": original_obj_act, "relationship": item["relationship"]}

    return {
        "act_identification": {
            "correctly_identified_acts": sorted([original_case_golden.get(act, original_case_extracted.get(act, act)) for act in correctly_identified_acts]),
            "false_positive_acts": sorted([original_case_extracted.get(act, act) for act in false_positive_acts]),
            "false_negative_acts": sorted([original_case_golden.get(act, act) for act in false_negative_acts]),
            "counts": act_identification_results["counts"]
        },
        "correct_relationships": [restore_case({"object_act": obj, "relationship": rel}) for (obj, rel) in sorted(correct_relationships_set)],
        "false_positives": [restore_case({"object_act": obj, "relationship": rel}) for (obj, rel) in sorted(false_positives_set)],
        "false_negatives": [restore_case({"object_act": obj, "relationship": rel}) for (obj, rel) in sorted(false_negatives_set)],
        "counts": {
            "correct_relationships": len(correct_relationships_set),
            "false_positives": len(false_positives_set),
            "false_negatives": len(false_negatives_set),
            "extracted_relationships": len(all_extracted_relations_set),
            "golden_relationships": len(all_golden_relations_set)
        }
    }


def evaluate_all(
    all_extracted_relationships: Dict[str, Dict[str, List[str]]],
    golden_standard_path: str = "gold-standard-test-dataset.xlsx"
) -> Dict[str, Any]:
    """
    Evaluates all extracted relationships for all subject Acts against a golden standard dataset.

    Args:
        all_extracted_relationships (Dict[str, Dict[str, List[str]]]):
            A dictionary where keys are subject Act titles and values are dictionaries
            of extracted relationships for that Act.
        golden_standard_path (str): The path to the golden standard Excel file.

    Returns:
        Dict[str, Any]: A dictionary containing the aggregated verification results.
    """
    try:
        xls = pd.ExcelFile(golden_standard_path)
    except FileNotFoundError:
        return {"error": f"Golden standard file not found at: {golden_standard_path}"}

    all_golden_data: Dict[str, Dict[str, List[str]]] = {}
    sheet_titles: Dict[str, str] = {}  # sheet_name -> subject_act_title

    for sheet_name in xls.sheet_names:
        try:
            df_title = pd.read_excel(xls, sheet_name=sheet_name, usecols="A", header=None, nrows=1)
            if not df_title.empty and pd.notna(df_title.iloc[0, 0]):
                subject_act_title = df_title.iloc[0, 0].strip()
                sheet_titles[sheet_name] = subject_act_title

                df = pd.read_excel(xls, sheet_name=sheet_name)
                relationship_codes = [str(code) for code in df.columns[1:]]

                golden_data_for_act: Dict[str, List[str]] = {}
                for _, row in df.iterrows():
                    object_act_title = row.iloc[0]
                    if pd.isna(object_act_title):
                        continue

                    object_act_title = str(object_act_title).strip()
                    if object_act_title not in golden_data_for_act:
                        golden_data_for_act[object_act_title] = []

                    for i, code in enumerate(relationship_codes):
                        if pd.notna(row.iloc[i + 1]) and str(row.iloc[i + 1]).strip() in ['1', '1.0']:
                            golden_data_for_act[object_act_title].append(code)

                all_golden_data[subject_act_title] = golden_data_for_act
        except Exception:
            continue

    # Normalize all data consistently for comparison (match verify_relationships)
    # Use normalize_for_comparison for both subject and object Act titles.
    normalized_extracted = {}
    for sub, obj_rels in all_extracted_relationships.items():
        norm_sub = normalize_for_comparison(sub)
        if not norm_sub:
            continue
        norm_obj_rels = {}
        for obj, rels in obj_rels.items():
            norm_obj = normalize_for_comparison(obj)
            if not norm_obj:
                continue
            norm_obj_rels[norm_obj] = rels
        normalized_extracted[norm_sub] = norm_obj_rels

    normalized_golden = {}
    for sub, obj_rels in all_golden_data.items():
        norm_sub = normalize_for_comparison(sub)
        if not norm_sub:
            continue
        norm_obj_rels = {}
        for obj, rels in obj_rels.items():
            norm_obj = normalize_for_comparison(obj)
            if not norm_obj:
                continue
            norm_obj_rels[norm_obj] = rels
        normalized_golden[norm_sub] = norm_obj_rels

    # --- Relationship Identification ---
    def flatten_relationships(data: Dict[str, Dict[str, List[str]]]) -> set:
        relations = set()
        for sub, obj_rels in data.items():
            for obj, rels in obj_rels.items():
                for rel in rels:
                    relations.add((sub, obj, rel))
        return relations

    extracted_relations_flat = flatten_relationships(normalized_extracted)
    golden_relations_flat = flatten_relationships(normalized_golden)

    correct_relationships_flat = extracted_relations_flat.intersection(golden_relations_flat)
    false_positives_flat = extracted_relations_flat.difference(golden_relations_flat)
    false_negatives_flat = golden_relations_flat.difference(extracted_relations_flat)

    # --- Act Identification (Summation over all subject Acts) ---
    total_correct_acts = 0
    total_fp_acts = 0
    total_fn_acts = 0
    total_extracted_acts = 0
    total_golden_acts = 0
    
    all_correctly_identified_acts = []
    all_false_positive_acts = []
    all_false_negative_acts = []

    for subject_act_lower, golden_obj_acts in normalized_golden.items():
        extracted_obj_acts = normalized_extracted.get(subject_act_lower, {})
        
        golden_act_titles = set(golden_obj_acts.keys())
        extracted_act_titles = set(extracted_obj_acts.keys())

        correct_acts = golden_act_titles.intersection(extracted_act_titles)
        fp_acts = extracted_act_titles.difference(golden_act_titles)
        fn_acts = golden_act_titles.difference(extracted_act_titles)

        total_correct_acts += len(correct_acts)
        total_fp_acts += len(fp_acts)
        total_fn_acts += len(fn_acts)
        total_extracted_acts += len(extracted_act_titles)
        total_golden_acts += len(golden_act_titles)

        all_correctly_identified_acts.extend(list(correct_acts))
        all_false_positive_acts.extend(list(fp_acts))
        all_false_negative_acts.extend(list(fn_acts))


    # Restore original casing for reporting
    original_case_map = {}
    for sub, obj_rels in all_extracted_relationships.items():
        original_case_map[sub.lower()] = sub
        for obj in obj_rels.keys():
            original_case_map[obj.lower()] = obj

    for sub, obj_rels in all_golden_data.items():
        original_case_map.setdefault(sub.lower(), sub)
        for obj in obj_rels.keys():
            original_case_map.setdefault(obj.lower(), obj)

    def restore_case_relation(rel_tuple):
        sub, obj, rel = rel_tuple
        return {
            "subject_act": original_case_map.get(sub, sub),
            "object_act": original_case_map.get(obj, obj),
            "relationship": rel
        }

    return {
        "act_identification": {
            "correctly_identified_acts": sorted([original_case_map.get(act, act) for act in set(all_correctly_identified_acts)]),
            "false_positive_acts": sorted([original_case_map.get(act, act) for act in set(all_false_positive_acts)]),
            "false_negative_acts": sorted([original_case_map.get(act, act) for act in set(all_false_negative_acts)]),
            "counts": {
                "correct": total_correct_acts,
                "false_positives": total_fp_acts,
                "false_negatives": total_fn_acts,
                "extracted": total_extracted_acts,
                "golden": total_golden_acts,
            }
        },
        "relationship_identification": {
            "correct_relationships": sorted([restore_case_relation(r) for r in correct_relationships_flat], key=lambda x: (x['subject_act'], x['object_act'])),
            "false_positives": sorted([restore_case_relation(r) for r in false_positives_flat], key=lambda x: (x['subject_act'], x['object_act'])),
            "false_negatives": sorted([restore_case_relation(r) for r in false_negatives_flat], key=lambda x: (x['subject_act'], x['object_act'])),
            "counts": {
                "correct": len(correct_relationships_flat),
                "false_positives": len(false_positives_flat),
                "false_negatives": len(false_negatives_flat),
                "extracted": len(extracted_relations_flat),
                "golden": len(golden_relations_flat),
            }
        }
    }
