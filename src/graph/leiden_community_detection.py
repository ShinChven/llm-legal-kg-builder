"""
Run Leiden community detection using Neo4j Graph Data Science (GDS) and export results.

Overview
--------
- Projects an in-memory graph of `Act` nodes connected by selected relationships.
- Uses an undirected projection (typical for community detection).
- Runs `gds.leiden.stream` and exports node -> community assignments to Excel.

Relationship scope
------------------
- Prioritises the same relationship types used by PageRank (CITES/CIT, AMENDS/AMD,
  PARTIAL_REPEALS/PRP, REPEALS/FRP). If none of those exist in the DB, it falls back
  to including all relationship types between Act nodes.

Output
------
- Excel: outputs/analytics/leiden_communities_YYYYMMDD_HHMMSS.xlsx (sheet: "leiden")
  Columns: Act, community, communitySize [and intermediateCommunityIds when requested]
- Optional PNG (when using `--visual`): outputs/analytics/leiden_communities_YYYYMMDD_HHMMSS_3d_sphere.png
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

from src.graph.neo4j_connection import close_driver, neo4j_session
from src.db.db_connection import db_connection
from src.db.create_leiden_communities_table import ensure_leiden_communities_table
from psycopg2.extras import Json


# Relationship type synonyms (reused from PageRank script semantics)
SYNONYMS: Mapping[str, Sequence[str]] = {
    "CITES": ("CITES", "CIT"),
    "AMENDS": ("AMENDS", "AMD"),
    "PARTIAL_REPEALS": ("PARTIAL_REPEALS", "PRP"),
    "REPEALS": ("REPEALS", "FRP"),
}

SUPPORTED_REL_TYPES: Set[str] = {alias for aliases in SYNONYMS.values() for alias in aliases}


def _drop_graph(session, graph_name: str) -> None:
    """Drop an existing GDS graph if it exists (best-effort)."""
    try:
        rec = (
            session.run(
                """
                CALL gds.graph.exists($name)
                YIELD exists
                RETURN exists
                """,
                name=graph_name,
            ).single()
            or {}
        )
    except Exception:
        return
    if rec.get("exists"):
        session.run(
            """
            CALL gds.graph.drop($name, false)
            YIELD graphName
            RETURN graphName
            """,
            name=graph_name,
        )


def _discover_rel_types_limited(session) -> Set[str]:
    """Return subset of supported (synonym) relationship types present in DB."""
    result = session.run(
        """
        MATCH ()-[r]->()
        RETURN DISTINCT type(r) AS type
        """
    )
    types_in_db = {rec["type"] for rec in result if rec and rec.get("type")}
    return {t for t in SUPPORTED_REL_TYPES if t in types_in_db}


def _discover_all_rel_types_between_acts(session) -> Set[str]:
    """Return all relationship types that connect Act nodes (fallback)."""
    result = session.run(
        """
        MATCH (:Act)-[r]->(:Act)
        RETURN DISTINCT type(r) AS type
        """
    )
    return {rec["type"] for rec in result if rec and rec.get("type")}


def _project_graph(
    session,
    graph_name: str,
    rel_types: Sequence[str],
    *,
    relationship_weight_property: Optional[str] = None,
) -> None:
    """Project an undirected graph of Act nodes using given relationship types.

    If `relationship_weight_property` is provided, aggregate weights by SUM; otherwise unweighted.
    """
    _drop_graph(session, graph_name)

    if relationship_weight_property:
        rel_proj = {
            t: {
                "type": t,
                "orientation": "UNDIRECTED",
                "properties": {
                    relationship_weight_property: {
                        "property": relationship_weight_property,
                        "aggregation": "SUM",
                    }
                },
            }
            for t in rel_types
        }
        session.run(
            """
            CALL gds.graph.project(
                $graph_name,
                ['Act'],
                $rel_proj
            )
            """,
            graph_name=graph_name,
            rel_proj=rel_proj,
        ).consume()
    else:
        # Build per-type projection to enforce UNDIRECTED orientation
        rel_proj = {
            t: {
                "type": t,
                "orientation": "UNDIRECTED",
            }
            for t in rel_types
        }
        session.run(
            """
            CALL gds.graph.project(
                $graph_name,
                ['Act'],
                $rel_proj
            )
            """,
            graph_name=graph_name,
            rel_proj=rel_proj,
        ).consume()


def _stream_leiden(
    session,
    *,
    graph_name: str,
    resolution: float,
    max_iter: int,
    tol: float,
    include_intermediate: bool = False,
    relationship_weight_property: Optional[str] = None,
) -> pd.DataFrame:
    """Run Leiden in stream mode and return DataFrame with Act, status, community id and size.

    This implementation is resilient to GDS version differences: it will remove
    unsupported configuration keys and retry automatically.
    """

    # Start with a superset of commonly supported keys
    config: Dict[str, object] = {
        "includeIntermediateCommunities": bool(include_intermediate),
    }
    # Add optional keys that may or may not be available depending on GDS version
    # We'll drop them if the server reports them as unexpected.
    config_candidates: Dict[str, object] = {
        "resolution": float(resolution),
        "maxIterations": int(max_iter),
        "tolerance": float(tol),
    }
    config.update(config_candidates)
    if relationship_weight_property:
        config["relationshipWeightProperty"] = relationship_weight_property

    # Helper to try a call and strip unknown keys if the server rejects them
    def try_leiden_call(algo_name: str, cfg: Dict[str, object]):
        if include_intermediate:
            query = (
                f"CALL {algo_name}($graph_name, $config) "
                "YIELD nodeId, communityId, intermediateCommunityIds "
                "RETURN gds.util.asNode(nodeId).title AS Act, "
                "coalesce(gds.util.asNode(nodeId).status, 'unknown') AS status, "
                "communityId AS community, intermediateCommunityIds AS intermediateCommunityIds "
                "ORDER BY community ASC, Act ASC"
            )
            cols = ["Act", "status", "community", "intermediateCommunityIds"]
        else:
            query = (
                f"CALL {algo_name}($graph_name, $config) "
                "YIELD nodeId, communityId "
                "RETURN gds.util.asNode(nodeId).title AS Act, "
                "coalesce(gds.util.asNode(nodeId).status, 'unknown') AS status, "
                "communityId AS community "
                "ORDER BY community ASC, Act ASC"
            )
            cols = ["Act", "status", "community"]

        try:
            rows = session.run(query, graph_name=graph_name, config=cfg).data()
            return pd.DataFrame(rows, columns=cols), None
        except Exception as e:  # Will parse for unknown keys
            msg = str(e)
            marker = "Unexpected configuration keys:"
            if marker in msg:
                unknown_part = msg.split(marker, 1)[1]
                # e.g. " maxIterations, resolution.}"
                unknown_part = unknown_part.strip()
                # cut at first closing brace or newline/period
                for end_token in ["}", ")", "\n"]:
                    if end_token in unknown_part:
                        unknown_part = unknown_part.split(end_token, 1)[0]
                # remove trailing period
                unknown_part = unknown_part.rstrip(". ")
                unknown_keys = [k.strip() for k in unknown_part.split(",") if k.strip()]
                return None, unknown_keys
            raise

    # Try with stable endpoint first, then beta
    algo_variants = ["gds.leiden.stream", "gds.beta.leiden.stream"]

    df: Optional[pd.DataFrame] = None
    remaining_config = dict(config)
    for algo in algo_variants:
        cfg = dict(remaining_config)
        for _ in range(5):  # up to a few pruning passes
            result, unknown = try_leiden_call(algo, cfg)
            if result is not None:
                df = result
                break
            # remove unknown keys and retry
            if unknown:
                for k in unknown:
                    cfg.pop(k, None)
            else:
                break
        if df is not None:
            break

    if df is None:
        # Final attempts: minimal config, then empty config
        minimal = {"includeIntermediateCommunities": bool(include_intermediate)}
        if relationship_weight_property:
            minimal["relationshipWeightProperty"] = relationship_weight_property
        try:
            rows = session.run(
                "CALL gds.leiden.stream($graph_name, $config) "
                "YIELD nodeId, communityId "
                "RETURN gds.util.asNode(nodeId).title AS Act, "
                "coalesce(gds.util.asNode(nodeId).status, 'unknown') AS status, "
                "communityId AS community "
                "ORDER BY community ASC, Act ASC",
                graph_name=graph_name,
                config=minimal,
            ).data()
            df = pd.DataFrame(rows, columns=["Act", "status", "community"])
        except Exception:
            rows = session.run(
                "CALL gds.leiden.stream($graph_name, {}) "
                "YIELD nodeId, communityId "
                "RETURN gds.util.asNode(nodeId).title AS Act, "
                "coalesce(gds.util.asNode(nodeId).status, 'unknown') AS status, "
                "communityId AS community "
                "ORDER BY community ASC, Act ASC",
                graph_name=graph_name,
            ).data()
            df = pd.DataFrame(rows, columns=["Act", "status", "community"])

    # Attach sizes
    if "community" in df.columns:
        sizes = df.groupby("community").size().rename("communitySize")
        df = df.join(sizes, on="community")
    return df


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _save_leiden_results_to_db(df: pd.DataFrame, include_intermediate: bool = False) -> None:
    """Persist Leiden results to PostgreSQL.

    Behavior:
    - Ensures the destination table exists (idempotent).
    - Truncates the table on each run.
    - Inserts one row per Act with community assignment and size.
    """
    # If DB pool isn't initialised, skip gracefully
    if not hasattr(db_connection, "get_connection"):
        print("PostgreSQL connection not available. Skipped saving Leiden results to SQL.")
        return

    # Ensure destination table exists
    ensure_leiden_communities_table()

    conn = db_connection.get_connection()
    if conn is None:
        print("PostgreSQL connection not available. Skipped saving Leiden results to SQL.")
        return

    try:
        with conn.cursor() as cur:
            # Truncate before inserting fresh results and reset the SERIAL id
            cur.execute("TRUNCATE TABLE leiden_communities RESTART IDENTITY;")

            has_intermediate = bool(include_intermediate and "intermediateCommunityIds" in df.columns)

            rows = []
            for _, row in df.iterrows():
                act_title = str(row.get("Act")) if pd.notna(row.get("Act")) else None
                status = str(row.get("status")) if pd.notna(row.get("status")) else None
                community = int(row.get("community")) if pd.notna(row.get("community")) else None
                community_size = int(row.get("communitySize")) if pd.notna(row.get("communitySize")) else None
                intermediate = row.get("intermediateCommunityIds") if has_intermediate else None

                rows.append(
                    (
                        act_title,
                        status,
                        community,
                        community_size,
                        Json(intermediate) if intermediate is not None else None,
                    )
                )

            if rows:
                cur.executemany(
                    """
                    INSERT INTO leiden_communities
                        (act_title, status, community, community_size, intermediate_community_ids)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    rows,
                )

        conn.commit()
        print(f"Saved {len(rows)} Leiden community rows to table 'leiden_communities' (truncated before insert).")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error saving Leiden results to database: {e}")
    finally:
        if conn:
            db_connection.release_connection(conn)


def _visualize_communities_3d_sphere(
    session,
    df: pd.DataFrame,
    output_path: str,
    rel_types: Set[str]
) -> str:
    """
    Create a 3D sphere network visualization with communities clustered together.

    Args:
        session: Neo4j session for querying relationships
        df: DataFrame with columns: Act, community, communitySize
        output_path: Path for the output image
        rel_types: Set of relationship types to include

    Returns:
        Path to the saved visualization
    """
    print("\n" + "="*60)
    print("Starting 3D Sphere Network Visualization")
    print("="*60)

    # Step 1: Build NetworkX graph
    print("\n[Step 1/7] Building NetworkX graph from Neo4j data...")
    G = nx.Graph()

    # Add all nodes from DataFrame
    for _, row in df.iterrows():
        G.add_node(row['Act'], community=int(row['community']), size=int(row['communitySize']))
    print(f"  ✓ Added {G.number_of_nodes()} nodes")

    # Query relationships from Neo4j
    print(f"\n[Step 2/7] Querying relationships from Neo4j...")
    rel_types_str = ', '.join([f'"{t}"' for t in rel_types])
    query = f"""
    MATCH (a1:Act)-[r]->(a2:Act)
    WHERE type(r) IN [{rel_types_str}]
    AND a1.title IN $acts AND a2.title IN $acts
    RETURN a1.title AS source, a2.title AS target, type(r) AS type
    """
    acts_list = df['Act'].tolist()
    result = session.run(query, acts=acts_list)

    edge_count = 0
    for record in result:
        G.add_edge(record['source'], record['target'])
        edge_count += 1
    print(f"  ✓ Added {edge_count} edges")

    # Step 3: Analyze community structure
    print(f"\n[Step 3/7] Analyzing community structure...")
    community_info = df.groupby('community').agg({
        'Act': 'count',
        'communitySize': 'first'
    }).rename(columns={'Act': 'node_count'}).reset_index()
    community_info = community_info.sort_values('node_count', ascending=False)

    print(f"  ✓ Found {len(community_info)} communities")
    print(f"  - Largest community: {community_info.iloc[0]['node_count']} nodes")
    print(f"  - Smallest community: {community_info.iloc[-1]['node_count']} nodes")

    # Identify isolated nodes (degree 0)
    isolated_nodes = [node for node, degree in G.degree() if degree == 0]
    print(f"  - Isolated nodes: {len(isolated_nodes)}")

    # Step 4: Generate color palette
    print(f"\n[Step 4/7] Generating color palette for communities...")
    n_communities = len(community_info)

    # Generate dramatically different, vibrant colors with red, green, yellow, orange, blue, etc.
    def generate_vibrant_colors(n):
        """Generate vibrant, dramatically different colors."""
        # Hand-picked vibrant colors for maximum contrast and variety
        base_colors = [
            '#e41a1c',  # Bright Red
            '#377eb8',  # Bright Blue
            '#4daf4a',  # Bright Green
            '#ff7f00',  # Bright Orange
            '#ffff33',  # Bright Yellow
            '#a65628',  # Brown
            '#f781bf',  # Pink
            '#984ea3',  # Purple
            '#00ced1',  # Dark Turquoise
            '#006D5B',  # Lime Green
            '#ff1493',  # Deep Pink
            '#1e90ff',  # Dodger Blue
            '#ff4500',  # Orange Red
            '#9370db',  # Medium Purple
            '#00fa9a',  # Medium Spring Green
            '#ff6347',  # Tomato
            '#4169e1',  # Royal Blue
            '#ffd700',  # Gold
            '#dc143c',  # Crimson
            '#00ffff',  # Cyan
        ]

        # Convert hex to RGBA for the base colors
        colors = []
        for i in range(min(n, len(base_colors))):
            rgb = mcolors.hex2color(base_colors[i])
            colors.append(tuple(rgb) + (1.0,))

        # If we need more than 20, add HSV-based colors WITHOUT shuffling
        if n > len(base_colors):
            remaining = n - len(base_colors)
            for i in range(remaining):
                # Evenly space hues, high saturation for vibrant colors
                hue = (i / remaining) * 0.95
                saturation = 0.85 + 0.10 * (i % 2)
                value = 0.85 + 0.10 * ((i + 1) % 2)
                rgb = mcolors.hsv_to_rgb([hue, saturation, value])
                colors.append(tuple(rgb) + (1.0,))

        return np.array(colors)

    colors = generate_vibrant_colors(n_communities)

    # Map each community ID to its unique color (ordered by size: largest gets first color, etc.)
    # community_info is already sorted by node_count descending, so colors are assigned by size
    community_colors = {}
    for i, (_, row) in enumerate(community_info.iterrows()):
        comm_id = int(row['community'])
        node_count = int(row['node_count'])
        community_colors[comm_id] = colors[i]
        if i < 5:  # Print first 5 for verification
            color_name = ['Red', 'Blue', 'Green', 'Orange', 'Yellow'][i] if i < 5 else f'Color {i+1}'
            print(f"    Community {comm_id} ({node_count} nodes) → {color_name}")

    print(f"  ✓ Generated {len(community_colors)} vibrant, dramatically different colors")
    print(f"  - Colors assigned by community size: Largest = Red, 2nd = Blue, 3rd = Green, etc.")
    print(f"  - Color palette: red, blue, green, orange, yellow, brown, pink, purple, cyan, and more")

    # Step 5: Calculate 3D sphere positions
    print(f"\n[Step 5/7] Calculating 3D sphere positions...")
    pos_3d = {}

    # Separate large communities from isolated/small ones
    # Large community threshold is set to 10 nodes
    large_community_threshold = 10
    large_communities = community_info[community_info['node_count'] >= large_community_threshold]['community'].tolist()
    small_communities = community_info[community_info['node_count'] < large_community_threshold]['community'].tolist()

    print(f"  - Large communities (≥{large_community_threshold} nodes): {len(large_communities)}")
    print(f"  - Small communities (<{large_community_threshold} nodes): {len(small_communities)}")

    # Position large communities on the sphere
    sphere_radius = 10
    n_large = len(large_communities)

    # Use golden spiral for even distribution on sphere
    golden_ratio = (1 + 5**0.5) / 2
    angle_increment = 2 * np.pi / golden_ratio

    community_centers = {}
    for i, comm_id in enumerate(large_communities):
        # Golden spiral distribution
        theta = i * angle_increment
        phi = np.arccos(1 - 2 * (i + 0.5) / max(n_large, 1))

        x = sphere_radius * np.sin(phi) * np.cos(theta)
        y = sphere_radius * np.sin(phi) * np.sin(theta)
        z = sphere_radius * np.cos(phi)

        community_centers[comm_id] = np.array([x, y, z])

    print(f"  ✓ Positioned {len(large_communities)} large communities on sphere")

    # Position nodes within each large community
    for comm_id in large_communities:
        comm_nodes = df[df['community'] == comm_id]['Act'].tolist()
        n_nodes = len(comm_nodes)
        center = community_centers[comm_id]

        # Create tight cluster around center
        cluster_radius = 0.5 + 0.3 * np.log(n_nodes + 1)

        for j, node in enumerate(comm_nodes):
            # Use golden spiral within cluster
            theta = j * angle_increment
            phi = np.arccos(1 - 2 * (j + 0.5) / max(n_nodes, 1))

            offset_x = cluster_radius * np.sin(phi) * np.cos(theta)
            offset_y = cluster_radius * np.sin(phi) * np.sin(theta)
            offset_z = cluster_radius * np.cos(phi)

            pos_3d[node] = center + np.array([offset_x, offset_y, offset_z])

    # Position small communities and isolated nodes around the sphere
    outer_radius = sphere_radius * 1.5
    small_and_isolated = small_communities
    n_small = len(small_and_isolated)

    for i, comm_id in enumerate(small_and_isolated):
        theta = i * angle_increment
        phi = np.arccos(1 - 2 * (i + 0.5) / max(n_small, 1))

        x = outer_radius * np.sin(phi) * np.cos(theta)
        y = outer_radius * np.sin(phi) * np.sin(theta)
        z = outer_radius * np.cos(phi)

        comm_nodes = df[df['community'] == comm_id]['Act'].tolist()

        # Scatter nodes in small community
        for j, node in enumerate(comm_nodes):
            angle = j * 2 * np.pi / max(len(comm_nodes), 1)
            offset = 0.3 * np.array([np.cos(angle), np.sin(angle), 0])
            pos_3d[node] = np.array([x, y, z]) + offset

    print(f"  ✓ Positioned {len(small_communities)} small communities on outer sphere")

    # Position truly isolated nodes
    for node in isolated_nodes:
        if node not in pos_3d:
            # Random position on outer sphere
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)

            x = outer_radius * 1.2 * np.sin(phi) * np.cos(theta)
            y = outer_radius * 1.2 * np.sin(phi) * np.sin(theta)
            z = outer_radius * 1.2 * np.cos(phi)

            pos_3d[node] = np.array([x, y, z])

    print(f"  ✓ Total nodes positioned: {len(pos_3d)}")

    # Step 6: Create 3D visualization
    print(f"\n[Step 6/7] Creating 3D visualization...")
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')

    # Draw edges
    print(f"  - Drawing {G.number_of_edges()} edges...")
    for edge in G.edges():
        if edge[0] in pos_3d and edge[1] in pos_3d:
            x_coords = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
            y_coords = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
            z_coords = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
            ax.plot(x_coords, y_coords, z_coords, 'gray', alpha=0.15, linewidth=0.3)

    # Draw nodes
    print(f"  - Drawing {G.number_of_nodes()} nodes...")
    for node in G.nodes():
        if node in pos_3d:
            comm = G.nodes[node]['community']
            # Use light grey for isolated nodes, community color for others
            if node in isolated_nodes:
                color = "#cccccccc"  # Light grey
                size = 20
            else:
                color = community_colors[comm]
                size = 40
            ax.scatter(pos_3d[node][0], pos_3d[node][1], pos_3d[node][2],
                      c=[color], s=size, alpha=0.8, edgecolors='black', linewidths=0.5)

    # Create legend
    print(f"  - Creating legend...")
    legend_elements = []
    for _, row in community_info.iterrows():
        comm_id = int(row['community'])
        node_count = int(row['node_count'])
        color = community_colors[comm_id]
        label = f"Community {comm_id} ({node_count} nodes)"
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=8, label=label))

    # Show only top communities in legend to avoid clutter
    max_legend_items = 15
    if len(legend_elements) > max_legend_items:
        legend_elements = legend_elements[:max_legend_items]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='gray', markersize=8,
                                         label=f'... and {len(community_info) - max_legend_items} more'))

    # Add isolated nodes to legend if any exist
    if len(isolated_nodes) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='lightgrey', markersize=6,
                                         label=f'Isolated Nodes ({len(isolated_nodes)} nodes)'))

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1),
             fontsize=10, framealpha=0.9)

    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'3D Community Network Visualization\n{G.number_of_nodes()} Nodes, '
                f'{G.number_of_edges()} Edges, {len(community_info)} Communities',
                fontsize=14, fontweight='bold', pad=20)

    # Remove grid for cleaner look
    ax.grid(True, alpha=0.3)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    # Shorter axis limits to bring sphere closer to viewer (zoom in effect)
    max_range = sphere_radius * 1.3  # Reduced from 2.0 to 1.3 for closer view
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    print(f"  ✓ Visualization created")

    # Step 7: Save figure
    print(f"\n[Step 7/7] Saving visualization...")
    viz_path = output_path.replace('.xlsx', '_3d_sphere.png')
    ensure_parent_dir(viz_path)
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved to: {viz_path}")
    print("\n" + "="*60)
    print("3D Visualization Complete!")
    print("="*60 + "\n")

    return viz_path


def run(
    *,
    resolution: float = 1.0,
    max_iter: int = 10,
    tol: float = 1e-6,
    include_intermediate: bool = False,
    weighted: bool = False,
    visual: bool = False,
) -> str:
    """Project graph, run Leiden, and export results to Excel.

    Args:
        resolution: Modularity resolution parameter.
        max_iter:   Maximum passes of refinement.
        tol:        Convergence tolerance.
        include_intermediate: If True, includes intermediate communities (hierarchy).
        weighted:   If True, uses relationship property 'weight' if present.
    Returns:
        Output Excel file path.
    """
    with neo4j_session() as session:
        rel_types = _discover_rel_types_limited(session)
        if not rel_types:
            # Fallback to all types connecting Act nodes
            rel_types = _discover_all_rel_types_between_acts(session)
        if not rel_types:
            raise RuntimeError("No relationship types found to project for Leiden.")

        graph_name = "legislation_leiden"
        _project_graph(
            session,
            graph_name,
            sorted(rel_types),
            relationship_weight_property=("weight" if weighted else None),
        )
        try:
            df = _stream_leiden(
                session,
                graph_name=graph_name,
                resolution=resolution,
                max_iter=max_iter,
                tol=tol,
                include_intermediate=include_intermediate,
                relationship_weight_property=("weight" if weighted else None),
            )
        finally:
            _drop_graph(session, graph_name)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("outputs", "analytics", f"leiden_communities_{ts}.xlsx")
        ensure_parent_dir(out_path)

        # Create community summary (sorted by size, largest to smallest)
        community_summary = df.groupby('community').agg({
            'Act': 'count'
        }).rename(columns={'Act': 'node_count'}).reset_index()
        community_summary = community_summary.sort_values('node_count', ascending=False).reset_index(drop=True)
        community_summary['Community Name'] = community_summary['community'].apply(lambda x: f'Community {x}')
        community_summary = community_summary[['Community Name', 'node_count']]
        community_summary.columns = ['Community', 'Number of Nodes']

        # Calculate statistics for the statistics sheet
        # Large community threshold is set to 10 nodes
        large_community_threshold = 10

        large_communities_count = len(community_summary[community_summary['Number of Nodes'] >= large_community_threshold])
        small_communities_count = len(community_summary[community_summary['Number of Nodes'] < large_community_threshold])

        # Isolated communities are those with only 1 node
        isolated_communities_count = len(community_summary[community_summary['Number of Nodes'] == 1])

        # Build a temporary graph to count isolated nodes (nodes with no edges)
        # For efficiency, we'll query this from Neo4j
        isolated_nodes_query = """
        MATCH (a:Act)
        WHERE a.title IN $acts
        AND NOT EXISTS {
            MATCH (a)-[r]-(other:Act)
            WHERE type(r) IN $rel_types AND other.title IN $acts
        }
        RETURN count(a) AS isolated_count
        """
        rel_types_list = list(rel_types)
        acts_list = df['Act'].tolist()
        isolated_result = session.run(isolated_nodes_query, acts=acts_list, rel_types=rel_types_list)
        isolated_nodes_count = isolated_result.single()['isolated_count']

        # Create statistics dataframe
        statistics_data = {
            'Metric': [
                'Total Communities',
                'Large Communities (≥{} nodes)'.format(int(large_community_threshold)),
                'Small Communities (<{} nodes)'.format(int(large_community_threshold)),
                'Isolated Communities (1 node)',
                'Isolated Nodes (no edges)',
                'Total Nodes'
            ],
            'Count': [
                len(community_summary),
                large_communities_count,
                small_communities_count,
                isolated_communities_count,
                isolated_nodes_count,
                len(df)
            ]
        }
        statistics_df = pd.DataFrame(statistics_data)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            # Write main leiden data
            cols = ["Act", "community", "communitySize"]
            if include_intermediate and "intermediateCommunityIds" in df.columns:
                cols = cols + ["intermediateCommunityIds"]
            df[cols].to_excel(writer, sheet_name="leiden", index=False)

            # Write community summary
            community_summary.to_excel(writer, sheet_name="community_summary", index=False)

            # Write statistics
            statistics_df.to_excel(writer, sheet_name="statistics", index=False)

        print(f"Wrote Leiden communities to {out_path}")
        print(f"  - Sheet 'leiden': Detailed node assignments")
        print(f"  - Sheet 'community_summary': Communities sorted by size (largest to smallest)")
        print(f"  - Sheet 'statistics': Community and node statistics")

        # Save to SQL table (truncate then insert fresh results)
        _save_leiden_results_to_db(df, include_intermediate=include_intermediate)

        # Optionally create 3D sphere visualization
        if visual:
            viz_path = _visualize_communities_3d_sphere(session, df, out_path, rel_types)
            print(f"Created 3D visualization at {viz_path}")
        else:
            print("Visualization skipped (use --visual to generate image).")

    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Leiden community detection on Acts in Neo4j (GDS).")
    ap.add_argument("--resolution", type=float, default=1.0, help="Modularity resolution (default: 1.0)")
    ap.add_argument("--max-iter", type=int, default=10, help="Maximum iterations (default: 10)")
    ap.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance (default: 1e-6)")
    ap.add_argument(
        "--include-intermediate",
        action="store_true",
        help="Include intermediate community levels in the output ordering.",
    )
    ap.add_argument(
        "--weighted",
        action="store_true",
        help="Use relationship property 'weight' if present during projection and algorithm run.",
    )
    ap.add_argument(
        "--visual",
        action="store_true",
        help="Generate and save the 3D sphere community visualization PNG.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(
            resolution=args.resolution,
            max_iter=args.max_iter,
            tol=args.tol,
            include_intermediate=args.include_intermediate,
            weighted=args.weighted,
            visual=args.visual,
        )
    finally:
        close_driver()


if __name__ == "__main__":
    main()
