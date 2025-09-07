import numpy as np
import time
from environment import START_POS, GOAL_POS, OBSTACLES, plot_simulation

# --- Algorithm Constants ---
STEP_SIZE = 0.1         # How far the robot moves in each step
GOAL_THRESHOLD = 0.2    # How close to the goal we need to be
MAX_ITERATIONS = 60000  # A safety Threshold to prevent infinite loops

# --- State Identifiers ---
MOVE_TO_GOAL = "MOVE_TO_GOAL"
WALL_FOLLOW = "WALL_FOLLOW"

# --- Utility Functions ---

# This function calculates the euclidean distances between two points
def distance(p1, p2):

    return np.linalg.norm(np.array(p1) - np.array(p2))


# Checks if a point is inside any obstacle. Returns the collided obstacle or None.
def check_collision(point, obstacles):

    for obs in obstacles:
        if distance(point, obs['center']) <= obs['radius']:
            return obs
    return None

# calculate the resultant path length when a path is found
def calculate_path_length(path):

    length = 0.0
    for i in range(len(path) - 1):
        length += distance(path[i], path[i+1])
    return length

# --- Bug 0 Algorithm Implementation ---
def bug0_planner(verbose=True):
    """
    Implements the Bug 0 path planning algorithm.
    Returns:
        list: The sequence of points representing the generated path.
    """
    # Initialization
    current_pos = np.array(START_POS, dtype=float)
    goal_pos = np.array(GOAL_POS, dtype=float)
    path = [tuple(current_pos)]
    state = MOVE_TO_GOAL
    followed_obstacle = None
    
    for _ in range(MAX_ITERATIONS):
        # 1. Check for success
        if distance(current_pos, goal_pos) < GOAL_THRESHOLD:
            if verbose: print("A collision-free path is found")
            path.append(tuple(goal_pos))
            return path

        # 2. Execute state logic
        if state == MOVE_TO_GOAL:
            direction = (goal_pos - current_pos) / distance(current_pos, goal_pos)
            new_pos = current_pos + direction * STEP_SIZE
            
            collided_obs = check_collision(new_pos, OBSTACLES)
            if collided_obs:
                state = WALL_FOLLOW
                followed_obstacle = collided_obs
            else:
                current_pos = new_pos

        elif state == WALL_FOLLOW:
            # Check leave condition
            direction_to_goal = (goal_pos - current_pos) / distance(current_pos, goal_pos)
            lookahead_pos = current_pos + direction_to_goal * STEP_SIZE
            
            # If a small step towards the goal is clear of the current obstacle, leave wall-following
            if distance(lookahead_pos, followed_obstacle['center']) > followed_obstacle['radius']:
                state = MOVE_TO_GOAL
                followed_obstacle = None
                current_pos = lookahead_pos
            else:
                # Follow the wall (LEFT-Turn)
                vec_to_center = followed_obstacle['center'] - current_pos
                tangent_vec = np.array([-vec_to_center[1], vec_to_center[0]]) # Rotate 90 degrees
                tangent_vec /= np.linalg.norm(tangent_vec)
                
                # Move along the tangent
                current_pos += tangent_vec * STEP_SIZE
                
                # Project back to the circle boundary to prevent drifting
                vec_from_center = current_pos - followed_obstacle['center']
                current_pos = followed_obstacle['center'] + (vec_from_center / np.linalg.norm(vec_from_center)) * followed_obstacle['radius']
        
        path.append(tuple(current_pos))

    if verbose: print("Failure: Max iterations reached. Could not find the goal.")
    return path

# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- Bug 0 Algorithm ---")
    
    # To Measure computational time
    start_time = time.time()
    generated_path = bug0_planner()
    end_time = time.time()
    
    computation_time = end_time - start_time
    
    if generated_path:
        #  To Calculate path length
        path_len = calculate_path_length(generated_path)
        
        print("\n--- Results for Bug 0 Algorithm ---")
        print(f" (i) Path Length: {path_len:.2f} units")
        print(f"(ii) Computational Time: {computation_time:.4f} seconds")
        
        # Generate the plot
        plot_simulation(generated_path, algorithm_name="Bug 0")