import numpy as np
import random
import time
import heapq
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Importing th environment variables from the environment.py file
from environment import START_POS, GOAL_POS, OBSTACLES

# --- Constants ---
NUM_SAMPLES = 500
K_NEIGHBORS = 15
MAX_EDGE_LEN = 5.0

# --- UtilityFunctions ---

#This fucntion calculates distance between two nodes
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to check for collision
def is_collision(point, obstacles):
    for obs in obstacles:
        if distance(point, obs['center']) <= obs['radius']:
            return True
    return False
#function to check if an edge drawn between two sampled nodes lies or collides with any obstacle
def is_edge_in_collision(p1, p2, obstacles):
    samples = np.linspace(p1, p2, 10)
    for sample in samples:
        if is_collision(sample, obstacles):
            return True
    return False

# to calculate final path length
def calculate_path_length(path):
    length = 0.0
    for i in range(len(path) - 1):
        length += distance(path[i], path[i+1])
    return length

# dijkstra algo
def dijkstra_search(graph, start_node_idx, goal_node_idx):

    distances = {node: float('inf') for node in graph}
    distances[start_node_idx] = 0
    predecessor = {node: None for node in graph}
    pq = [(0, start_node_idx)]
    while pq:
        dist, current_node = heapq.heappop(pq)
        if dist > distances[current_node]:
            continue
        if current_node == goal_node_idx:
            break
        for neighbor, weight in graph.get(current_node, []):
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessor[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))
    path = []
    current = goal_node_idx
    while current is not None:
        path.append(current)
        current = predecessor[current]
    if distances[goal_node_idx] == float('inf'):
        return None, float('inf')
    return path[::-1], distances[goal_node_idx]


# --- PRM Algorithm Implementation ---

# The verbose variable is used to omit the print statements when performance_comparision.py 
def prm_planner(verbose=True):
    """
    Implements the Probabilistic Roadmap (PRM) path planning algorithm.
    """
    # Sampling randomly through uniform distribution
    if verbose: print(f"Sampling {NUM_SAMPLES} nodes...")
    nodes = []
    while len(nodes) < NUM_SAMPLES:
        sample = (random.uniform(0, 30), random.uniform(0, 30))
        if not is_collision(sample, OBSTACLES):
            nodes.append(sample)
    
    nodes.append(START_POS)
    start_node_idx = len(nodes) - 1
    nodes.append(GOAL_POS)
    goal_node_idx = len(nodes) - 1
    
    # Connecting Nodes
    # if verbose: print(f"Connecting nodes to their {K_NEIGHBORS} nearest neighbors...")
    kdtree = KDTree(nodes)
    graph = {i: [] for i in range(len(nodes))}

    for i, node in enumerate(nodes):
        distances, indices = kdtree.query(node, k=K_NEIGHBORS + 1)
        for j in range(1, len(indices)):
            neighbor_idx = indices[j]
            dist = distances[j]
            if dist > MAX_EDGE_LEN:
                continue
            if not is_edge_in_collision(node, nodes[neighbor_idx], OBSTACLES):
                graph[i].append((neighbor_idx, dist))
                graph[neighbor_idx].append((i, dist))
                
    # 3. Compute the shortest path between start node and goal node
    if verbose: print("Searching for the shortest path using Dijkstra's algorithm...")
    path_indices, path_length = dijkstra_search(graph, start_node_idx, goal_node_idx)
    
    if path_indices:
        final_path = [nodes[i] for i in path_indices]
        if verbose: print("Path found!")
        return final_path, (nodes, graph)
    else:
        if verbose: print("Path could not be found.")
        return None, (nodes, graph)

#  Plotting the Computed Path 
def plot_prm_result(path, roadmap):
    nodes, graph = roadmap
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(START_POS[0], START_POS[1], 'b*', markersize=15, label='Start')
    ax.plot(GOAL_POS[0], GOAL_POS[1], 'ro', markersize=15, markerfacecolor='none', markeredgewidth=3, label='Goal')
    for obs in OBSTACLES:
        ax.add_patch(patches.Circle(obs['center'], obs['radius'], edgecolor=obs['color'], facecolor='none', linewidth=2))
    node_coords = np.array(nodes)
    ax.plot(node_coords[:, 0], node_coords[:, 1], 'x', color='gray', markersize=5, label='PRM Nodes')
    for node_idx, neighbors in graph.items():
        for neighbor_idx, _ in neighbors:
            if node_idx < neighbor_idx:
                p1 = nodes[node_idx]
                p2 = nodes[neighbor_idx]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='lightgray', linewidth=0.7)
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'black', linewidth=3, label='PRM Path', zorder=10)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_title('Probabilistic Roadmap (PRM) Simulation')
    ax.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    print("--- Starting Probabilistic Roadmap (PRM) Planner ---")
    start_time = time.time()
    final_path, roadmap = prm_planner() 
    end_time = time.time()
    computation_time = end_time - start_time
    print("\n--- Performance Results for PRM ---")
    if final_path:
        path_len = calculate_path_length(final_path)
        print(f" (i) Path Length: {path_len:.2f} units")
    else:
        print(" (i) Path Length: Not applicable (no path found)")
    print(f"(ii) Computational Time: {computation_time:.4f} seconds")
    plot_prm_result(final_path, roadmap)