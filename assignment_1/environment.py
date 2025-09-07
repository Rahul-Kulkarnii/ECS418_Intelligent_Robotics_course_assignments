import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# --- Environment Configuration Data ---
START_POS = (1, 1)
GOAL_POS = (20, 20)
OBSTACLES = [
    {'center': (4.5, 3), 'radius': 2, 'color': 'goldenrod', 'label': 'Obstacle 1'},
    {'center': (3, 12), 'radius': 2, 'color': 'darkorchid', 'label': 'Obstacle 2'},
    {'center': (15, 15), 'radius': 3, 'color': 'forestgreen', 'label': 'Obstacle 3'}
]

def plot_simulation(path, algorithm_name="Bug 0"):
    """
    Plots the environment, obstacles, and the final path of the robot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot Start and Goal
    ax.plot(START_POS[0], START_POS[1], 'b*', markersize=15, label='Start')
    ax.plot(GOAL_POS[0], GOAL_POS[1], 'ro', markersize=15, markerfacecolor='none', markeredgewidth=2, label='Goal')

    # Plot Obstacles
    obstacle_legend_handles = []
    for obs in OBSTACLES:
        circle = patches.Circle(obs['center'], obs['radius'], edgecolor=obs['color'], facecolor='none', linewidth=2)
        ax.add_patch(circle)
        obstacle_legend_handles.append(Line2D([0], [0], color=obs['color'], lw=2, label=obs['label']))

    # Plot the calculated path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, color='black', linewidth=2, label=f'{algorithm_name} Path', zorder=10)

    # --- Final Plot Adjustments ---
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_title(f'{algorithm_name} Algorithm Simulation')

    # Create the complete legend
    point_legend_handles, point_labels = ax.get_legend_handles_labels()
    all_handles = point_legend_handles + obstacle_legend_handles
    ax.legend(handles=all_handles, loc='upper right')

    plt.show()