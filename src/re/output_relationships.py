import os
import csv
import argparse
from datetime import datetime

from src.re.act_relationship_handler import ActRelationshipHandler


# Default output directory (kept consistent with other tools under outputs/re)
OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "re", "exports")
)


def flatten_relationships(tree: dict, max_layers: int):
    """
    Traverse the relationship tree up to max_layers and yield flattened edges.

    Yields tuples of (subject_act, object_act, relationship_code_or_empty).
    If a child edge has multiple relationship codes, yields one record per code.
    If a child edge has no relationship codes, yields a single record with empty code.
    """
    if not tree or "act" not in tree:
        return

    queue = [(tree, 0)]  # (node, layer)
    while queue:
        node, layer = queue.pop(0)
        if layer >= max_layers:
            # Do not expand beyond requested depth
            continue

        subject = node.get("act")
        for child in node.get("children", []) or []:
            obj = child.get("act")
            rels = child.get("relationship", []) or []

            if rels:
                for code in rels:
                    yield (subject, obj, code)
            else:
                yield (subject, obj, "")

            # Enqueue child to continue layering
            queue.append((child, layer + 1))


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject Act", "Object Act", "Relationship"])  # header
        for row in rows:
            writer.writerow(row)


def write_markdown(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| Subject Act | Object Act | Relationship |\n")
        f.write("|---|---|---|\n")
        for subject, obj, rel in rows:
            # Escape pipe characters to keep table intact
            s = (subject or "").replace("|", "\\|")
            o = (obj or "").replace("|", "\\|")
            r = (rel or "").replace("|", "\\|")
            f.write(f"| {s} | {o} | {r} |\n")


def main(title: str, layers: int):
    handler = ActRelationshipHandler()

    print(f"Fetching {layers} layer(s) of relationships for '{title}'...")
    tree = handler.get_relationship_layers(title, layers)

    if not tree or not tree.get("children"):
        print("No relationship data found.")
        return

    flattened = list(flatten_relationships(tree, layers))

    # Build filenames
    safe_title = title.replace(" ", "_").replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{safe_title}_L{layers}_{ts}"
    csv_path = os.path.join(OUTPUT_DIR, f"{base}.csv")
    md_path = os.path.join(OUTPUT_DIR, f"{base}.md")

    # Write outputs
    write_csv(flattened, csv_path)
    write_markdown(flattened, md_path)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Output relationships for an Act as CSV and Markdown."
    )
    parser.add_argument("title", type=str, help="The title of the Act.")
    parser.add_argument(
        "layers", type=int, help="Number of relationship layers to include."
    )
    args = parser.parse_args()

    main(args.title, args.layers)

