import argparse
import os
import yaml
from datetime import datetime
from src.re.act_relationship_handler import ActRelationshipHandler

def pull_relationships(title: str, layers: int):
    """
    Pulls a specified number of layers of relationships for an Act and saves it to a file.

    :param title: The title of the Act to start with.
    :param layers: The number of relationship layers to pull.
    """
    handler = ActRelationshipHandler()

    print(f"Pulling {layers} layer(s) of relationships for '{title}'...")

    data = handler.get_relationship_layers(title, layers)

    if not data:
        print("No relationships found.")
        return

    # Ensure output directory exists
    output_dir = './outputs/re/layers/'
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize title for filename
    safe_title = title.replace(' ', '_').replace('/', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_{layers}layers_{timestamp}.yaml"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully saved relationship data to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull relationship layers for a given Act.")
    parser.add_argument("title", type=str, help="The title of the Act.")
    parser.add_argument("layers", type=int, help="The number of layers to pull.")

    args = parser.parse_args()

    pull_relationships(args.title, args.layers)
