import time
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg') # FIX: Explicitly set an interactive backend (e.g., TkAgg, Qt5Agg)
import matplotlib.pyplot as plt
NUM_RUNS = 100
# Import the planner functions from your other scripts
try:
    # We call the planners but don't need their plotting functions here
    from bug0_planner import bug0_planner, calculate_path_length as bug_len
    from prm_planner import prm_planner, calculate_path_length as prm_len
    from rrt_planner import rrt_planner, calculate_path_length as rrt_len
except ImportError as e:
    print(f"Error: Make sure all planner files are in the same directory.")
    print(f"Details: {e}")
    sys.exit(1)


def plot_performance_results(summary_data):
    """
    Generates and displays bar plots for performance comparison.
    """
    algorithms = list(summary_data.keys())
    avg_times = [data['avg_time'] for data in summary_data.values()]
    std_times = [data['std_time'] for data in summary_data.values()]
    avg_lengths = [data['avg_len'] for data in summary_data.values()]
    std_lengths = [data['std_len'] for data in summary_data.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Plot 1: Average Computation Time ---
    bars1 = ax1.bar(algorithms, avg_times, yerr=std_times, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Average Computation Time', fontsize=14)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_xticklabels(algorithms, rotation=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.bar_label(bars1, fmt='%.3f')

    # --- Plot 2: Average Path Length ---
    bars2 = ax2.bar(algorithms, avg_lengths, yerr=std_lengths, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Average Path Length', fontsize=14)
    ax2.set_ylabel('Path Length (units)', fontsize=12)
    ax2.set_xticklabels(algorithms, rotation=15)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.bar_label(bars2, fmt='%.2f')

    fig.suptitle('Plannnig Algorithm Performance Comparison (100 Runs)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def run_performance_test():
    """
    Runs each path planning algorithm multiple times to gather and display performance data.
    """
    
    
    results = {
        "Bug 0": {"times": [], "lengths": []},
        "PRM":   {"times": [], "lengths": []},
        "RRT":   {"times": [], "lengths": []}
    }
    
    print(f"--- Starting Performance Test (Running each algorithm {NUM_RUNS} times) ---")
    
    for i in range(NUM_RUNS):
        print(f"\n--- Run {i + 1}/{NUM_RUNS} ---")
        
        # --- Test Algorithms (verbose is False to keep console clean) ---
        print("Running Bug 0...")
        start_time = time.time()
        bug_path = bug0_planner(verbose=False)
        comp_time = time.time() - start_time
        if bug_path:
            results["Bug 0"]["times"].append(comp_time)
            results["Bug 0"]["lengths"].append(bug_len(bug_path))

        print("Running PRM...")
        start_time = time.time()
        prm_path, _ = prm_planner(verbose=False)
        comp_time = time.time() - start_time
        if prm_path:
            results["PRM"]["times"].append(comp_time)
            results["PRM"]["lengths"].append(prm_len(prm_path))

        print("Running RRT...")
        start_time = time.time()
        rrt_path, _ = rrt_planner(verbose=False)
        comp_time = time.time() - start_time
        if rrt_path:
            results["RRT"]["times"].append(comp_time)
            results["RRT"]["lengths"].append(rrt_len(rrt_path))
            
    # --- Calculate and Print Averages ---
    print("\n\n--- Final Performance Comparison ---")
    print("-" * 75)
    header = f"{'Algorithm':<15} | {'Avg. Time (s)':<15} | {'Std. Dev. (Time)':<15} | {'Avg. Path Length':<20} | {'Std. Dev. (Length)':<15}"
    print(header)
    print("-" * 75)
    
    summary_data = {}
    for name, data in results.items():
        avg_time = np.mean(data["times"])
        std_time = np.std(data["times"])
        avg_len = np.mean(data["lengths"])
        std_len = np.std(data["lengths"])
        
        summary_data[name] = {
            'avg_time': avg_time, 'std_time': std_time,
            'avg_len': avg_len, 'std_len': std_len
        }
        
        print(f"{name:<15} | {avg_time:<15.4f} | {std_time:<15.4f} | {avg_len:<20.2f} | {std_len:<15.2f}")
    
    print("-" * 75)
    return summary_data


if __name__ == '__main__':
    final_summary = run_performance_test()
    if final_summary:
        print("\nDisplaying performance plots...")
        plot_performance_results(final_summary)