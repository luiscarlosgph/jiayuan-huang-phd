from Utils import *


if __name__ == "__main__":
    # Example 1: Compare Random vs Active under different pool sizes
    # plot_random_vs_active_by_pool([100, 500, 1000, 5000, 10000], k=None)
    
    # Example 2: Compare Random vs Active under different iteration counts
    # plot_random_vs_active_by_iterations([5, 10, 20, 30, 40, 50], k=None)
    
    # Example 3: Fixed total budget, compare different batch sizes k
    # plot_random_vs_active_by_k([1, 2, 5, 10, 20, 50], total_budget=1000, n_iterations=10)
    
    # Example 4: Random -> Active switch analysis under different budgets
    plot_random_to_active_switch_by_budget([50, 100, 200, 400], n_iterations=10)
    
    # Example 5: Sweep different pool sizes with a specific strategy
    # sweep_and_plot(N_pool_list=[100, 500, 1000, 5000, 10000], strategy='active', k=None)
    
    # Example 6: Sweep different iteration counts with a specific strategy
    # sweep_and_plot(N_iterations_list=[5, 10, 20, 30, 40, 50], strategy='random', k=None)
