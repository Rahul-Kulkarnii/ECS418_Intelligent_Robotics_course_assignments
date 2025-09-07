import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import environment variables from the environment.py 
from environment import START_POS, GOAL_POS, OBSTACLES

# --- Algorithm Constants ---
MAX_ITERATIONS = 10000
STEP_SIZE = 0.5        
GOAL_BIAS = 0.1


# --- RRT Node Class ---
# A node in the RRT tree, storing coordinates and a reference to its parent
class Node:
    
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos, dtype=float)
        self.parent = parent

# Utility Functions

# distance between two nodes
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# checking condition for collision with obstacles
def is_collision(point, obstacles):
    for obs in obstacles:
        if distance(point, obs['center']) <= obs['radius']:
            return True
    return False

# edge inside obstacle?
def is_edge_in_collision(p1, p2, obstacles):
    samples = np.linspace(p1, p2, 10)
    for sample in samples:
        if is_collision(sample, obstacles):
            return True
    return False


def get_nearest_node(tree, random_pos):
    min_dist = float('inf')
    nearest_node = None
    for node in tree:
        dist = distance(node.pos, random_pos)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node


def steer(from_node, to_pos, step_size):
    direction = to_pos - from_node.pos
    dist = np.linalg.norm(direction)
    if dist == 0:
        return Node(from_node.pos, parent=from_node)
    direction /= dist  # Normalize
    new_pos = from_node.pos + direction * min(step_size, dist)
    return Node(new_pos, parent=from_node)

#total path lenght result
def calculate_path_length(path):
    length = 0.0
    for i in range(len(path) - 1):
        length += distance(path[i], path[i+1])
    return length

# --- RRT Algorithm Implementation (Modified) ---
def rrt_planner(verbose=True):
    tree = [Node(START_POS)]
    goal_pos_arr = np.array(GOAL_POS, dtype=float)
    
    for i in range(MAX_ITERATIONS):
        # Sample a random point, with a bias towards the goal
        if random.random() < GOAL_BIAS:
            random_pos = goal_pos_arr
        else:
            random_pos = np.array([random.uniform(0, 30), random.uniform(0, 30)])
            
        # Find the nearest node and steer towards the random point
        nearest_node = get_nearest_node(tree, random_pos)
        new_node = steer(nearest_node, random_pos, STEP_SIZE)
        
        # If the new branch is collision-free, add the new node to the tree
        if not is_edge_in_collision(nearest_node.pos, new_node.pos, OBSTACLES):
            tree.append(new_node)
            
            # Check if the new node is exactly at the goal.
            # This happens when the goal is sampled and is within STEP_SIZE of the nearest node.
            if np.array_equal(new_node.pos, goal_pos_arr):
                if verbose: print(f"Goal reached after {i+1} iterations!")
                
                # Reconstruct the path by backtracking from the goal node ie reverse the array
                path = []
                current = new_node
                while current is not None:
                    path.append(tuple(current.pos))
                    current = current.parent
                return path[::-1], tree
                    
    if verbose: print(f"Failure: Could not find a path to the goal within {MAX_ITERATIONS} iterations.")
    return None, tree

# --- Plotting path ------
def plot_rrt_result(path, tree):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(START_POS[0], START_POS[1], 'b*', markersize=15, label='Start')
    ax.plot(GOAL_POS[0], GOAL_POS[1], 'ro', markersize=15, markerfacecolor='none', markeredgewidth=2, label='Goal')
    for obs in OBSTACLES:
        ax.add_patch(patches.Circle(obs['center'], obs['radius'], edgecolor=obs['color'], facecolor='none', linewidth=2))
    for node in tree:
        if node.parent:
            ax.plot([node.pos[0], node.parent.pos[0]], 
                    [node.pos[1], node.parent.pos[1]], 
                    '-', color='lightgray', linewidth=2)
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'black', linewidth=3, label='RRT Path', zorder=10)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_title('Rapidly-exploring Random Tree (RRT) Algorithm')
    ax.legend(loc='upper right')
    plt.show()
 

# main function 
if __name__ == '__main__':
    print("--- Starting RRT Planner ---")
    start_time = time.time()
    final_path, rrt_tree = rrt_planner()
    end_time = time.time()
    computation_time = end_time - start_time
    print("\n--- Performance Results for RRT ---")
    if final_path:
        path_len = calculate_path_length(final_path)
        print(f" (i) Path Length: {path_len:.2f} units")
    else:
        print(" (i) Path Length: Not applicable (no path found)")
    print(f"(ii) Computational Time: {computation_time:.4f} seconds")
    plot_rrt_result(final_path, rrt_tree)