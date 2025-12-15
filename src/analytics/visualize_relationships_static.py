import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from src.re.count_relationships_all import load_relationship_adjacency

# --- Constants to Edit ---
CORE_ACT_TITLE = "Employment Relations Act 2000"
NUMBER_OF_LAYERS = 2
# -------------------------

# Color Palette for layers
LAYER_COLORS = [
    '#FF6347',  # Tomato
    '#4682B4',  # SteelBlue
    '#32CD32',  # LimeGreen
    '#FFD700',  # Gold
    '#6A5ACD',  # SlateBlue
    '#FF4500',  # OrangeRed
    '#20B2AA',  # LightSeaGreen
    '#9370DB',  # MediumPurple
    '#DAA520',  # GoldenRod
    '#D2691E',  # Chocolate
    # Additional colors to extend palette to 20
    '#DC143C',  # Crimson
    '#1E90FF',  # DodgerBlue
    '#3CB371',  # MediumSeaGreen
    '#FF8C00',  # DarkOrange
    '#00CED1',  # DarkTurquoise
    '#C71585',  # MediumVioletRed
    '#8B4513',  # SaddleBrown
    '#00BFFF',  # DeepSkyBlue
    '#2F4F4F',  # DarkSlateGray
    '#228B22'   # ForestGreen
]

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 're', 'visualization', 'static'))

def _build_tree_from_adjacency(adj, start_act_title: str, num_layers: int) -> dict:
    """
    Build a relationship tree using an in-memory adjacency without per-node DB queries.

    The structure mirrors ActRelationshipHandler.get_relationship_layers output:
      { "act": <title>, "children": [ { "act": <child>, "relationship": [codes], "children": [...] }, ... ] }
    """
    if num_layers <= 0 or not start_act_title:
        return {}

    tree = {"act": start_act_title, "children": []}
    queue = [(start_act_title, tree, 0)]  # (current_subject, parent_node, current_layer)
    visited = {start_act_title}

    print(f"Collecting relationships (BFS) from '{start_act_title}' up to {num_layers} layer(s)...")

    current_layer_marker = 0
    processed_subjects_in_layer = 0
    edges_to_next = 0
    relcodes_to_next = 0
    new_acts_next = 0
    total_edges_considered = 0
    total_relcodes_seen = 0

    while queue:
        current_subject, parent_node, current_layer = queue.pop(0)
        if current_layer >= num_layers:
            continue

        # If we are moving to a new layer (due to BFS ordering), emit a summary for the previous one
        if current_layer != current_layer_marker:
            print(
                f"Layer {current_layer_marker} -> {current_layer_marker + 1}: "
                f"subjects {processed_subjects_in_layer}, edges {edges_to_next}, "
                f"new acts {new_acts_next}, relationship codes {relcodes_to_next}"
            )
            current_layer_marker = current_layer
            processed_subjects_in_layer = 0
            edges_to_next = 0
            relcodes_to_next = 0
            new_acts_next = 0

        for obj, rels in adj.get(current_subject, []) or []:

            relationship_types = rels or []
            child_node = {
                "act": obj,
                "relationship": relationship_types,
                "children": []
            }
            parent_node.setdefault("children", []).append(child_node)

            was_new = obj not in visited
            if was_new:
                visited.add(obj)
                queue.append((obj, child_node, current_layer + 1))

            # Progress accounting for this transition to next layer
            edges_to_next += 1
            total_edges_considered += 1
            rc = len(relationship_types) if relationship_types else 1
            relcodes_to_next += rc
            total_relcodes_seen += rc
            if was_new:
                new_acts_next += 1

        # Finished processing this subject in its layer
        processed_subjects_in_layer += 1

    # prune empty children arrays for cleaner traversal/printing
    def prune_empty_children(node):
        if "children" in node and node["children"]:
            for child in list(node["children"]):
                prune_empty_children(child)
            if not node["children"]:
                del node["children"]
        elif "children" in node:
            del node["children"]

    # Emit final layer summary if any processing occurred
    print(
        f"Layer {current_layer_marker} -> {current_layer_marker + 1}: "
        f"subjects {processed_subjects_in_layer}, edges {edges_to_next}, "
        f"new acts {new_acts_next}, relationship codes {relcodes_to_next}"
    )

    prune_empty_children(tree)
    print(
        f"Finished BFS. Visited {len(visited)} act(s). "
        f"Total edges considered {total_edges_considered}, total relationship codes {total_relcodes_seen}."
    )
    return tree

def create_combined_montage(image_paths, titles, output_filename, cols: int = 3):
    """Create a grid montage from a list of image paths and save to output_filename.

    - Uses matplotlib only (no extra dependencies).
    - Titles are shown above each panel when provided.
    """
    if not image_paths:
        return None

    import math
    import matplotlib.pyplot as plt

    n = len(image_paths)
    cols = max(1, min(cols, n))
    rows = int(math.ceil(n / cols))

    # Heuristic figure size: ~5.5x3.7 inches per panel to reduce padding footprint
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 3.7))
    axes = (axes if isinstance(axes, (list, tuple)) else [axes])
    # Flatten axes for uniform indexing
    import numpy as np
    axes = np.array(axes).reshape(-1)

    for idx, (path, title) in enumerate(zip(image_paths, titles)):
        ax = axes[idx]
        try:
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_axis_off()
            if title:
                ax.set_title(title, fontsize=14, pad=6)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image\n{os.path.basename(path)}\n{e}",
                    ha='center', va='center', fontsize=8)
            ax.set_axis_off()

    # Hide any remaining axes if grid > images
    for ax in axes[n:]:
        ax.set_visible(False)

    # Reduce inter-panel spacing and outer margins
    try:
        fig.tight_layout(pad=0.3)
    except Exception:
        pass
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, wspace=0.06, hspace=0.12)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fig.savefig(output_filename, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved combined montage to {output_filename}")
    return output_filename

def create_networkx_graph(data, output_filename, root_title, relationship_filter=None):
    """
    Creates and saves a static network graph visualization using NetworkX and Matplotlib.
    """
    # Use MultiDiGraph to support multiple parallel edges per pair
    G = nx.MultiDiGraph()
    node_colors_map = {}
    queue = [(data, 0)]  # Queue stores tuples of (node_data, layer)
    processed_nodes = set()

    # --- 1. Build the graph with NetworkX ---
    while queue:
        node, layer = queue.pop(0)
        subject_name = node.get("act")

        if subject_name in processed_nodes:
            continue
        processed_nodes.add(subject_name)

        # Always add the node and assign color based on layer (original behavior)
        G.add_node(subject_name)
        node_colors_map[subject_name] = LAYER_COLORS[layer % len(LAYER_COLORS)]

        if "children" in node:
            for child in node["children"]:
                object_name = child.get("act")
                relationships = child.get("relationship", []) or []

                # Always enqueue to explore deeper layers (original behavior)
                if object_name not in processed_nodes:
                    child_layer = layer + 1
                    queue.append((child, child_layer))

                # Add one edge per relationship code (supports multiple parallel edges),
                # honoring the original filtering behavior (keep unlabeled edges even when filtering)
                for rel_code in relationships if relationships else [None]:
                    if relationship_filter and rel_code is not None and rel_code not in relationship_filter:
                        continue
                    G.add_edge(subject_name, object_name)

    if not G.nodes:
        print("Graph is empty, skipping visualization.")
        return

    # --- 2. Configure plot settings ---
    plt.figure(figsize=(18, 18))
    node_count = len(G.nodes)

    # Use a spring layout for a network-style visualization
    # More iterations for larger graphs to spread nodes out
    iterations = 200 if node_count > 100 else 50
    pos = nx.spring_layout(G, iterations=iterations, seed=42)

    # Get the list of colors in the same order as the nodes
    node_color_list = [node_colors_map.get(node, '#000000') for node in G.nodes()]

    # --- 3. Draw the graph ---
    # Draw nodes (no labels by request)
    if node_count > 150:
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_color_list)
        edge_alpha = 0.4
    else:
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_color_list)
        edge_alpha = 0.6

    # Draw edges, showing multiple parallel edges with slight curvature differences
    # Group edge counts by (u, v)
    edge_multiplicity = {}
    for u, v, k in G.edges(keys=True):
        edge_multiplicity[(u, v)] = edge_multiplicity.get((u, v), 0) + 1

    # For each pair, draw each parallel edge with a distinct arc radius
    for (u, v), count in edge_multiplicity.items():
        if count == 1:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrows=True, alpha=edge_alpha, connectionstyle='arc3,rad=0.0')
        else:
            # Symmetric radii around 0: e.g., for 3 edges -> [-0.2, 0.0, 0.2]
            base = 0.2
            offsets = [base * (i - (count - 1) / 2) for i in range(count)]
            for rad in offsets:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrows=True, alpha=edge_alpha, connectionstyle=f'arc3,rad={rad}')

    # --- 4. Save the plot ---
    # Add a descriptive title that includes filter context (increase font size)
    if relationship_filter:
        title_suffix = ", ".join(sorted(relationship_filter))
        plt.title(f"Network Graph for {root_title} [{title_suffix}]", fontsize=24, pad=8)
    else:
        plt.title(f"Network Graph for {root_title} [comprehensive]", fontsize=24, pad=8)
    plt.axis('off')

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    # Tighten layout and reduce outer padding when saving
    try:
        plt.tight_layout(pad=0.2)
    except Exception:
        pass
    plt.savefig(output_filename, format="PNG", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close() # Close the figure to free memory

    print(f"Saved NetworkX graph to {output_filename}")
    return output_filename


def main(start_act_title, num_layers=3):
    """
    Main function to generate and save relationship graphs as static images.
    """
    print(f"Building {num_layers} layer(s) for '{start_act_title}' using in-memory adjacency (no per-node queries)...")

    # Load all relationships once into memory, then BFS in memory only
    try:
        adjacency = load_relationship_adjacency()
    except Exception as e:
        print(f"Failed to load adjacency from DB: {e}")
        return

    if not adjacency:
        print("Adjacency is empty; no relationships available. Aborting visualization.")
        return

    # Adjacency stats
    subjects = len(adjacency)
    edges_total = sum(len(v) for v in adjacency.values())
    relcodes_total = 0
    for lst in adjacency.values():
        for _, rels in lst:
            relcodes_total += len(rels) if rels else 1
    print(
        f"Loaded adjacency: {subjects} subject(s), {edges_total} edge row(s), "
        f"~{relcodes_total} relationship code(s)."
    )

    relationship_data = _build_tree_from_adjacency(adjacency, start_act_title, num_layers)

    if not relationship_data or not relationship_data.get("children"):
        print("No relationship data found.")
        return

    # --- 1. Create the comprehensive network graph ---
    comprehensive_output_path = os.path.join(
        OUTPUT_DIR,
        f"{start_act_title.replace(' ', '_')}_L{num_layers}_comprehensive.png"
    )
    create_networkx_graph(relationship_data, comprehensive_output_path, start_act_title)

    # --- 2. Create separate graphs for each relationship type ---
    all_relationship_types = set()
    def collect_relationship_types(node):
        if "children" in node:
            for child in node["children"]:
                for rel_type in child.get("relationship", []):
                    all_relationship_types.add(rel_type)
                collect_relationship_types(child)

    collect_relationship_types(relationship_data)
    print(f"Found relationship types: {all_relationship_types}")

    # Generate all per-type images and collect into a map
    type_to_path = {}
    for rel_type in sorted(all_relationship_types):
        filtered_output_path = os.path.join(
            OUTPUT_DIR,
            f"{start_act_title.replace(' ', '_')}_L{num_layers}_{rel_type}.png"
        )
        create_networkx_graph(relationship_data, filtered_output_path, start_act_title, relationship_filter=[rel_type])
        type_to_path[str(rel_type)] = filtered_output_path

    # --- 3. Create a combined montage in the requested order ---
    # Order: comprehensive first, then CIT, AMD, PRP, FRP (skip missing)
    montage_paths = [comprehensive_output_path]
    montage_titles = ["comprehensive"]
    preferred_order = ["CIT", "AMD", "PRP", "FRP"]
    for key in preferred_order:
        if key in type_to_path:
            montage_paths.append(type_to_path[key])
            montage_titles.append(key)

    if montage_paths:
        combined_output_path = os.path.join(
            OUTPUT_DIR,
            f"{start_act_title.replace(' ', '_')}_L{num_layers}_combined.png"
        )
        create_combined_montage(montage_paths, montage_titles, combined_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Act relationships as a static graph")
    parser.add_argument("title", nargs="?", default=CORE_ACT_TITLE, help="Core Act title (default: constant)")
    parser.add_argument("layers", nargs="?", type=int, default=NUMBER_OF_LAYERS, help="Number of layers to explore")
    args = parser.parse_args()

    main(args.title, args.layers)
