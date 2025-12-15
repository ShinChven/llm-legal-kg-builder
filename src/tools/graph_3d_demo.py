import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Parameters
group_count = 7
nodes_per_group = [7, 8, 6, 10, 9, 5, 8]  # Number of nodes in each cluster
colors = [
    '#e57373', '#81c784', '#64b5f6', '#ffd54f', '#ba68c8', '#4dd0e1', '#b0bec5'
]
entry_act_colors = [
    '#f44336', '#43a047', '#1976d2', '#ffb300', '#8e24aa', '#0097a7', '#546e7a'
]
entry_act_labels = [f'Entry Act {i+1}' for i in range(group_count)]
shared_act_label = 'Shared Act 0'

# Create graph
G = nx.Graph()

# Add shared act node
G.add_node(shared_act_label, group='shared', color='#444444')

# Add entry acts and connect to shared act
entry_act_nodes = []
for i in range(group_count):
    entry_act = entry_act_labels[i]
    G.add_node(entry_act, group='entry', color=entry_act_colors[i])
    G.add_edge(shared_act_label, entry_act)
    entry_act_nodes.append(entry_act)

# Add cluster nodes and connect to entry acts
cluster_nodes = []
for i, entry_act in enumerate(entry_act_nodes):
    for j in range(nodes_per_group[i]):
        node_label = f'{entry_act}_N{j+1}'
        G.add_node(node_label, group=f'cluster_{i}', color=colors[i])
        G.add_edge(entry_act, node_label)
        cluster_nodes.append(node_label)

# 3D layout: arrange entry acts in a circle, clusters as spheres around entry acts

# Spread out clusters even more (double previous)
angle_step = 2 * np.pi / group_count
radius = 40  # Increased from 36
center = np.array([0, 0, 0])

pos = {}
# Shared act at center
pos[shared_act_label] = center

# Entry acts in a circle
entry_positions = []
for i in range(group_count):
    angle = i * angle_step
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = 0
    pos[entry_act_labels[i]] = np.array([x, y, z])
    entry_positions.append(np.array([x, y, z]))

# Cluster nodes as spheres around entry acts
for i, entry_act in enumerate(entry_act_labels):
    entry_pos = pos[entry_act]
    
    # Calculate direction from center to entry act to push cluster outward
    # Since z is 0 for entry acts, we can just normalize x,y
    v_len = np.sqrt(entry_pos[0]**2 + entry_pos[1]**2 + entry_pos[2]**2)
    if v_len > 0:
        dir_vec = entry_pos / v_len
    else:
        dir_vec = np.array([1, 0, 0])
        
    # Push the center of the cluster sphere away from the entry act
    push_distance = 18
    cluster_center = entry_pos + dir_vec * push_distance

    n = nodes_per_group[i]
    phi = np.linspace(0, np.pi, n)
    theta = np.linspace(0, 2 * np.pi, n)
    for j in range(n):
        r = 10
        # Use cluster_center instead of entry_pos
        x = cluster_center[0] + r * np.sin(phi[j]) * np.cos(theta[j])
        y = cluster_center[1] + r * np.sin(phi[j]) * np.sin(theta[j])
        z = cluster_center[2] + r * np.cos(phi[j])
        node_label = f'{entry_act}_N{j+1}'
        pos[node_label] = np.array([x, y, z])

# Draw 3D graph
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw edges
for u, v in G.edges():
    x = [pos[u][0], pos[v][0]]
    y = [pos[u][1], pos[v][1]]
    z = [pos[u][2], pos[v][2]]
    ax.plot(x, y, z, color='#cccccc', alpha=0.7, linewidth=2)

# Draw nodes
text_z_offset = 5  # Lift labels above the nodes
for node in G.nodes():
    p = pos[node]
    color = G.nodes[node]['color']
    if G.nodes[node]['group'] == 'shared':
        ax.scatter(*p, s=1350, c=color, edgecolors='k', zorder=10)  # Double 300
        ax.text(p[0], p[1], p[2] + text_z_offset, node, fontsize=16, ha='center', va='center', weight='bold', color='white', zorder=20, bbox=dict(facecolor='#444444', edgecolor='none', boxstyle='round,pad=0.3'))
    elif G.nodes[node]['group'] == 'entry':
        ax.scatter(*p, s=900, c=color, edgecolors='k', zorder=9)  # Double 200
        ax.text(p[0], p[1], p[2] + text_z_offset, node, fontsize=15, ha='center', va='center', weight='bold', color='white', zorder=19, bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))
    else:
        ax.scatter(*p, s=360, c=color, edgecolors='k', alpha=0.8, zorder=8)  # Double 80

# Set limits and remove axes


ax.set_axis_off()
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_zlim(-30, 30)
first_graph_title = 'Multi-entry Concurrent Relationship Extraction'
first_graph_filename = first_graph_title.lower().replace(' ', '_').replace('-', '') + '.png'
ax.set_title(first_graph_title, fontsize=16, pad=20)
plt.tight_layout()

# Save image
plt.savefig(f'outputs/{first_graph_filename}', dpi=200, bbox_inches='tight')
plt.close()
print(f'3D graph image saved to outputs/{first_graph_filename}')

# --- New Simulation: Core Act expanding to 3 layers ---
print("-" * 30)
print("Generating Layered Graph Simulation...")

# Parameters for Layered Graph
layers_config = [
    {'count': 1, 'color': '#D50000', 'size': 2000, 'radius': 0, 'label': 'Core Act', 'legend_label': 'Core Act'}, # Layer 0
    {'count': 6, 'color': '#FF6D00', 'size': 1000, 'radius': 20, 'legend_label': 'Layer 1 Acts'}, # Layer 1
    {'count': 15, 'color': '#FFD600', 'size': 400, 'radius': 35, 'legend_label': 'Layer 2 Acts'}, # Layer 2
    {'count': 40, 'color': '#00C853', 'size': 100, 'radius': 50, 'legend_label': 'Layer 3 Acts'}  # Layer 3
]

G2 = nx.Graph()

# Generate Nodes and Edges
# Layer 0
core_id = "Core"
G2.add_node(core_id, layer=0, color=layers_config[0]['color'], size=layers_config[0]['size'])

# Layer 1
l1_ids = []
for i in range(layers_config[1]['count']):
    nid = f"L1_{i}"
    G2.add_node(nid, layer=1, color=layers_config[1]['color'], size=layers_config[1]['size'])
    G2.add_edge(core_id, nid)
    l1_ids.append(nid)

# Layer 2
l2_ids = []
for i in range(layers_config[2]['count']):
    nid = f"L2_{i}"
    G2.add_node(nid, layer=2, color=layers_config[2]['color'], size=layers_config[2]['size'])
    # Connect to a random L1
    parent = l1_ids[i % len(l1_ids)] # Distribute evenly
    G2.add_edge(parent, nid)
    l2_ids.append(nid)

# Layer 3
l3_ids = []
for i in range(layers_config[3]['count']):
    nid = f"L3_{i}"
    G2.add_node(nid, layer=3, color=layers_config[3]['color'], size=layers_config[3]['size'])
    # Connect to a random L2
    parent = l2_ids[i % len(l2_ids)] # Distribute evenly
    G2.add_edge(parent, nid)
    l3_ids.append(nid)

# Positioning
pos2 = {}
pos2[core_id] = np.array([0, 0, 0])

# L1 Placement (Fibonacci Sphere)
def fibonacci_sphere(samples, radius):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y) * radius
        theta = phi * i
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append(np.array([x, y * radius, z]))
    return points

l1_points = fibonacci_sphere(len(l1_ids), layers_config[1]['radius'])
for i, nid in enumerate(l1_ids):
    pos2[nid] = l1_points[i]

# L2 and L3 Placement (Directional Jitter)
def place_children(child_ids, graph, radius, randomness=0.3):
    for child in child_ids:
        # Find parent
        parents = list(graph.neighbors(child))
        # Assuming tree structure mostly, picking the first parent from lower layer
        # Note: edges are undirected, but we know the construction order
        # We look for the neighbor that is in the previous layer
        current_layer = graph.nodes[child]['layer']
        parent = None
        for p in parents:
            if graph.nodes[p]['layer'] == current_layer - 1:
                parent = p
                break
        
        if parent is None: # Should not happen by construction
            continue

        parent_pos = pos2[parent]
        
        # Direction vector
        if np.linalg.norm(parent_pos) == 0:
            direction = np.random.rand(3) - 0.5
        else:
            direction = parent_pos / np.linalg.norm(parent_pos)
        
        # Add some noise to direction
        noise = np.random.randn(3) * randomness
        new_direction = direction + noise
        new_direction = new_direction / np.linalg.norm(new_direction)
        
        pos2[child] = new_direction * radius

place_children(l2_ids, G2, layers_config[2]['radius'], randomness=0.5)
place_children(l3_ids, G2, layers_config[3]['radius'], randomness=0.5)

# Draw
fig2 = plt.figure(figsize=(12, 12))
ax2 = fig2.add_subplot(111, projection='3d')

# Edges
for u, v in G2.edges():
    p1 = pos2[u]
    p2 = pos2[v]
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#888888', alpha=0.4, linewidth=1)

# Nodes
for layer_idx in range(4):
    # Filter nodes by layer
    nodes_in_layer = [n for n, d in G2.nodes(data=True) if d.get('layer') == layer_idx]
    if not nodes_in_layer:
        continue
        
    xs = [pos2[n][0] for n in nodes_in_layer]
    ys = [pos2[n][1] for n in nodes_in_layer]
    zs = [pos2[n][2] for n in nodes_in_layer]
    
    conf = layers_config[layer_idx]
    ax2.scatter(xs, ys, zs, c=conf['color'], s=conf['size'], edgecolors='white', alpha=0.9, label=conf['legend_label'])
    
    if layer_idx == 0:
        ax2.text(xs[0], ys[0], zs[0]+5, conf['label'], fontsize=14, ha='center', weight='bold', zorder=100)

# Create legend with uniform marker sizes
leg = ax2.legend(loc='upper left', bbox_to_anchor=(0.8, 1.0), fontsize=12)
# Matplotlib 3.7+ uses legend_handles, older versions use legendHandles
handles = getattr(leg, 'legend_handles', getattr(leg, 'legendHandles', []))
for handle in handles:
    handle.set_sizes([300]) # Set fixed size for legend markers

ax2.dist = 7  # Zoom in (default is ~10)
ax2.set_axis_off()
ax2.set_xlim(-55, 55)
ax2.set_ylim(-55, 55)
ax2.set_zlim(-55, 55)
second_graph_title = 'In-depth Relationship Extraction'
second_graph_filename = second_graph_title.lower().replace(' ', '_').replace('-', '') + '.png'
ax2.set_title(second_graph_title, fontsize=16)
plt.tight_layout()

# Save image
plt.savefig(f'outputs/{second_graph_filename}', dpi=200, bbox_inches='tight')
plt.close()
print(f'Layered 3D graph image saved to outputs/{second_graph_filename}')