import os
import argparse
import math
from typing import Dict, Iterable, List, Tuple, Any

import matplotlib.pyplot as plt
import networkx as nx
from multiprocessing import Pool, cpu_count

from src.re.count_relationships_all import load_relationship_adjacency
import matplotlib.image as mpimg
from matplotlib.patches import Patch


# Defaults (aligned with visualize_relationships_static.py)
CORE_ACT_TITLE = "Employment Relations Act 2000"
NUMBER_OF_LAYERS = 2

# Colors per layer (reused palette)
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


EVOLUTION_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'analytics', 'evolution')
)


def build_layered_graph(
    adj: Dict[str, List[Tuple[str, Iterable[str]]]],
    start_act_title: str,
    num_layers: int,
    *,
    only_cit: bool = False,
) -> Tuple[nx.DiGraph, Dict[str, int]]:
    """
    Build a directed graph (edges collapsed, ignoring relationship types) using BFS layers.

    Returns (G, node_layer) where node_layer assigns the minimal discovery layer for each node
    with root at layer 0. Edges are included only between nodes discovered up to num_layers.
    If only_cit is True, only adjacency rows containing 'CIT' are used.
    """
    if not start_act_title or num_layers <= 0:
        return nx.DiGraph(), {}

    G = nx.DiGraph()
    node_layer: Dict[str, int] = {start_act_title: 0}
    visited = {start_act_title}

    # Initialize
    G.add_node(start_act_title)
    frontier: List[str] = [start_act_title]
    current_layer = 0

    print(f"BFS from '{start_act_title}' up to {num_layers} layer(s)" + (" [CIT only]" if only_cit else ""))

    while frontier and current_layer < num_layers:
        next_frontier: List[str] = []
        current_layer += 1

        for u in frontier:
            for v, rels in adj.get(u, []) or []:
                # Optionally limit to 'CIT' edges
                if only_cit and (not rels or 'CIT' not in set(rels)):
                    continue

                # Create edge, collapsing multiplicity by using DiGraph
                G.add_node(v)
                G.add_edge(u, v)

                # Discovery on first sight to assign minimal layer
                if v not in visited:
                    visited.add(v)
                    node_layer[v] = current_layer
                    next_frontier.append(v)

        frontier = next_frontier

    # Ensure all nodes have a layer assignment (should hold by construction)
    for n in G.nodes:
        if n not in node_layer:
            # Fallback: if something slipped through, pin to last known layer
            node_layer[n] = min(current_layer, num_layers)

    print(
        f"Built graph with {G.number_of_nodes()} node(s), {G.number_of_edges()} edge(s). "
        f"Max discovered layer: {max(node_layer.values()) if node_layer else 0}."
    )
    return G, node_layer


def _draw_subgraph(
    G_full: nx.DiGraph,
    node_layer: Dict[str, int],
    upto_layer: int,
    pos: Dict[str, Tuple[float, float]],
    *,
    output_path: str,
    title: str,
):
    """
    Draw a cumulative subgraph containing nodes with layer <= upto_layer using
    a shared layout `pos` computed from the full graph.
    """
    nodes = [n for n, l in node_layer.items() if l <= upto_layer]
    H = G_full.subgraph(nodes).copy()

    if not H.nodes:
        print(f"Layer {upto_layer}: empty subgraph, skipping.")
        return

    # Prepare colors per node based on their layer
    colors = [LAYER_COLORS[node_layer[n] % len(LAYER_COLORS)] for n in H.nodes]

    plt.figure(figsize=(16, 16))
    node_count = H.number_of_nodes()
    if node_count > 200:
        node_size = 40
        alpha = 0.35
    elif node_count > 120:
        node_size = 80
        alpha = 0.45
    elif node_count > 60:
        node_size = 160
        alpha = 0.55
    else:
        node_size = 400
        alpha = 0.65

    nx.draw_networkx_nodes(H, pos, node_size=node_size, node_color=colors)
    nx.draw_networkx_edges(H, pos, arrows=True, alpha=alpha)
    plt.title(title, fontsize=32, pad=14)
    plt.axis('off')

    # Add legend labeling each layer with its color
    try:
        layers_present = sorted({node_layer[n] for n in H.nodes})
        handles = [
            Patch(facecolor=LAYER_COLORS[l % len(LAYER_COLORS)], edgecolor='none', label=f"Layer {l}")
            for l in layers_present
        ]
        if handles:
            leg = plt.legend(
                handles=handles,
                title="Layers",
                loc="upper right",
                framealpha=0.85,
                fontsize=12,
                title_fontsize=12,
            )
            # Improve legend readability slightly
            leg.get_frame().set_linewidth(0.0)
    except Exception:
        # If legend creation fails, continue without blocking rendering
        pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    except Exception:
        pass
    plt.savefig(output_path, format="PNG", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def _draw_layer_job(args: Tuple[Any, Dict[str, int], int, Dict[str, Tuple[float, float]], str, str]) -> str:
    """Worker wrapper to draw a single layer frame in a separate process."""
    G_full, node_layer, upto_layer, pos, output_path, title = args
    _draw_subgraph(G_full, node_layer, upto_layer, pos, output_path=output_path, title=title)
    return output_path


def render_evolution(
    start_act_title: str,
    num_layers: int,
    *,
    only_cit: bool = False,
    output_dir: str = EVOLUTION_OUTPUT_DIR,
    jobs: int = 1,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Render a sequence of cumulative images for layers 1..num_layers using a stable layout.
    Returns a tuple of (list of image paths created, node_layer mapping).
    """
    # Load adjacency once
    try:
        adjacency = load_relationship_adjacency()
    except Exception as e:
        raise RuntimeError(f"Failed to load adjacency from DB: {e}")

    if not adjacency:
        print("Adjacency is empty; nothing to render.")
        return [], {}

    G_full, node_layer = build_layered_graph(
        adjacency, start_act_title, num_layers, only_cit=only_cit
    )
    if G_full.number_of_nodes() == 0:
        print("Empty graph; nothing to render.")
        return [], node_layer

    # Print counts per layer before rendering
    from collections import Counter
    node_counts = Counter(node_layer.values())

    edge_counts_per_layer = Counter()
    for u, v in G_full.edges():
        layer = max(node_layer.get(u, -1), node_layer.get(v, -1))
        if layer != -1:
            edge_counts_per_layer[layer] += 1

    print("\nNodes and edges per layer:")
    for i in range(num_layers + 1):
        nodes = node_counts.get(i, 0)
        edges = edge_counts_per_layer.get(i, 0)
        print(f"Layer {i}: {nodes} nodes, {edges} edges")
    print("")


    # Compute a stable layout using the full graph so frames align visually
    total_nodes = G_full.number_of_nodes()
    iterations = 200 if total_nodes > 120 else 80
    pos = nx.spring_layout(G_full, iterations=iterations, seed=42)

    # Ensure the position dict is plain-serializable (tuples), helpful for multiprocessing
    pos = {n: (float(p[0]), float(p[1])) for n, p in pos.items()}

    safe_title = start_act_title.replace(' ', '_')
    suffix = "_CIT" if only_cit else ""

    tasks: List[Tuple[Any, Dict[str, int], int, Dict[str, Tuple[float, float]], str, str]] = []
    for k in range(1, num_layers + 1):
        out_path = os.path.join(output_dir, f"{safe_title}_evo_L{k}{suffix}.png")
        frame_title = f"{start_act_title} – Evolution up to Layer {k}{' [CIT only]' if only_cit else ''}"
        tasks.append((G_full, node_layer, k, pos, out_path, frame_title))

    outputs: List[str] = []
    jobs = max(1, int(jobs or 1))
    if jobs == 1:
        print("Rendering in single-process mode.")
        for t in tasks:
            outputs.append(_draw_layer_job(t))
    else:
        num_cpus = cpu_count()
        print(f"[DEBUG] Requested jobs: {jobs}")
        print(f"[DEBUG] Available CPU cores: {num_cpus}")
        max_procs = min(jobs, num_cpus)
        print(f"Rendering {len(tasks)} frame(s) using {max_procs} process(es)...")
        with Pool(processes=max_procs) as pool:
            for out in pool.imap_unordered(_draw_layer_job, tasks):
                outputs.append(out)
    return sorted(outputs), node_layer


def combine_frames_to_grid(
    image_paths: List[str],
    out_path: str,
    *,
    cols: int = 3,
    dpi: int = 300,
    last_col_image_path: str = None,
) -> str:
    """
    Combine multiple image files into a single grid image and save to out_path.
    Keeps things dependency-light by using matplotlib only.
    """
    # Filter to existing files
    paths = [p for p in image_paths if p and os.path.isfile(p)]
    if not paths:
        print("No images to combine; skipping grid rendering.")
        return ""

    # Optionally append a final tile (e.g., histogram) just once at the end
    if last_col_image_path and os.path.isfile(last_col_image_path):
        paths = paths + [last_col_image_path]

    # Read first image to estimate size
    try:
        sample = mpimg.imread(paths[0])
    except Exception as e:
        print(f"Failed to read sample image for sizing: {e}")
        return ""

    if sample is None or len(sample.shape) < 2:
        print("Sample image invalid; cannot compute grid sizing.")
        return ""

    h, w = sample.shape[0], sample.shape[1]
    n = len(paths)
    cols = max(1, int(cols or 1))
    rows = math.ceil(n / cols)
    tile_paths = paths[:]

    # Figure size in inches, try to preserve per-tile pixel density
    fig_w = (w * cols) / dpi
    fig_h = (h * rows) / dpi

    # Render grid
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    # Normalize axes to 2D array for indexing
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis('off')
            if idx < len(tile_paths):
                p = tile_paths[idx]
                if p:
                    try:
                        img = mpimg.imread(p)
                        ax.imshow(img)
                    except Exception as e:
                        print(f"Failed to read image '{p}': {e}")
            idx += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        plt.tight_layout(pad=0.0)
    except Exception:
        pass
    fig.savefig(out_path, format="PNG", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Saved combined grid -> {out_path}")
    return out_path


def save_layer_histogram(node_layer: Dict[str, int], output_path: str, *, dpi: int = 300) -> str:
    """
    Save a bar chart image showing, for each discovered layer, the number of nodes.
    X-axis: layer number. Y-axis: number of nodes in that layer.
    Returns the saved image path.
    """
    if not node_layer:
        print("No node-layer data; skipping histogram rendering.")
        return ""

    from collections import Counter
    counts = Counter(node_layer.values())
    max_layer = max(counts.keys()) if counts else 0
    layers = list(range(0, max_layer + 1))
    values = [counts.get(i, 0) for i in layers]

    # Choose a square figure to roughly match evolution frames.
    plt.figure(figsize=(8, 8), dpi=dpi)
    # Color each bar according to its layer color
    bar_colors = [LAYER_COLORS[i % len(LAYER_COLORS)] for i in layers]
    bars = plt.bar(layers, values, color=bar_colors, edgecolor='black', linewidth=0.6)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Nodes', fontsize=14)
    plt.title('Nodes per Layer', fontsize=18, pad=10)

    # Color x-axis tick labels to match layer colors
    ax = plt.gca()
    ax.set_xticks(layers)
    ax.set_xticklabels([str(i) for i in layers], fontsize=12)
    for idx, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(bar_colors[idx])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='PNG', bbox_inches='tight')
    plt.close()
    print(f"Saved histogram -> {output_path}")
    return output_path


def compute_full_node_layers(start_act_title: str, *, only_cit: bool = False) -> Dict[str, int]:
    """
    Explore the network from the start node to exhaustion (no layer cap) and
    return a dict mapping node -> minimal discovery layer.
    """
    try:
        adjacency = load_relationship_adjacency()
    except Exception as e:
        raise RuntimeError(f"Failed to load adjacency from DB: {e}")

    if not start_act_title or not adjacency:
        return {}

    node_layer: Dict[str, int] = {start_act_title: 0}
    visited = {start_act_title}
    frontier: List[str] = [start_act_title]
    current_layer = 0

    while frontier:
        next_frontier: List[str] = []
        current_layer += 1
        for u in frontier:
            for v, rels in adjacency.get(u, []) or []:
                if only_cit and (not rels or 'CIT' not in set(rels)):
                    continue
                if v not in visited:
                    visited.add(v)
                    node_layer[v] = current_layer
                    next_frontier.append(v)
        frontier = next_frontier

    return node_layer


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize relationship evolution from a core Act. Generates cumulative frames for layers 1..N.\n"
            "Edges are treated as citations without type rendering; optionally filter to CIT only."
        )
    )
    parser.add_argument("title", nargs="?", default=CORE_ACT_TITLE, help="Core Act title (default: constant)")
    parser.add_argument("layers", nargs="?", type=int, default=NUMBER_OF_LAYERS, help="Number of layers (N)")
    parser.add_argument("--only-cit", action="store_true", help="Restrict edges to 'CIT' relationships only")
    parser.add_argument(
        "--output-dir",
        default=EVOLUTION_OUTPUT_DIR,
        help=f"Directory for output images (default: {EVOLUTION_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for rendering (processes). Default: 1",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="After rendering frames, also write a combined grid image.",
    )
    parser.add_argument(
        "--combine-cols",
        type=int,
        default=3,
        help="Number of columns in the combined grid (default: 3)",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help=(
            "Also render a layer histogram (layer number vs. node count). When used with --combine, "
            "the histogram is appended as the final tile in the combined grid."
        ),
    )
    args = parser.parse_args()

    print(
        f"Rendering evolution for '{args.title}' over {args.layers} layer(s)"
        + (" [CIT only]" if args.only_cit else "")
        + f" -> {args.output_dir}"
    )

    outputs, node_layer = render_evolution(
        args.title,
        args.layers,
        only_cit=args.only_cit,
        output_dir=args.output_dir,
        jobs=args.jobs,
    )

    hist_path = ""
    if args.histogram:
        # Compute histogram from the full reachable network (ignore layer cap)
        full_node_layer = compute_full_node_layers(args.title, only_cit=args.only_cit)
        if full_node_layer:
            safe_title = args.title.replace(' ', '_')
            suffix = "_CIT" if args.only_cit else ""
            hist_path = os.path.join(args.output_dir, f"{safe_title}_evo_hist{suffix}.png")
            save_layer_histogram(full_node_layer, hist_path)

    if args.combine and outputs:
        safe_title = args.title.replace(' ', '_')
        suffix = "_CIT" if args.only_cit else ""
        combined_path = os.path.join(
            args.output_dir, f"{safe_title}_evo_combined{suffix}.png"
        )
        combine_frames_to_grid(
            outputs,
            combined_path,
            cols=args.combine_cols,
            last_col_image_path=(hist_path if args.histogram and os.path.isfile(hist_path) else None),
        )


if __name__ == "__main__":
    main()
