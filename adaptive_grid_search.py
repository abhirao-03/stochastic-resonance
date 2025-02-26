import numpy as np
from optimiser import run
import matplotlib.pyplot as plt
from tqdm import tqdm

def adaptive_grid_search(time_hor,
    n_points: int = 10,
    min_jump_mult: float = 2.0,
    max_jump_mult: float = 4.0,
    samples_per_point: int = 5,
    n_refinements: int = 3,
    refinement_factor: float = 0.2,
    show_progress: bool = False,
    plot: bool = True):
    """
    Perform adaptive grid search over jump time multiplier with progressive refinement.
    
    Args:
        n_points: Number of grid points to evaluate in each iteration
        min_jump_mult: Initial minimum jump multiplier
        max_jump_mult: Initial maximum jump multiplier
        samples_per_point: Number of evaluations per grid point
        n_refinements: Number of refinement iterations
        refinement_factor: Factor to determine width of refined search region
        show_progress: Whether to show progress bars
        plot: Whether to show plots
        
    Returns:
        best_jump_mult: Overall best jump multiplier found
        best_pvalue: Best mean p-value found
        all_jump_mults: List of arrays of jump multipliers for each iteration
        all_mean_pvalues: List of arrays of mean p-values for each iteration
    """

    all_jump_mults = []
    all_mean_pvalues = []
    
    current_min = min_jump_mult
    current_max = max_jump_mult
    best_jump_mult = None
    best_pvalue = -np.inf
    
    for iteration in range(n_refinements + 1):
        print(f"\nIteration {iteration + 1}/{n_refinements + 1}")
        print(f"Searching range: [{current_min:.3f}, {current_max:.3f}]")
        print()
        
        # Create grid points with geometric spacing
        jump_mults = np.geomspace(current_min, current_max, n_points)
        mean_pvalues = np.zeros(n_points)
        
        # Evaluate each grid point
        grid_iterator = tqdm(enumerate(jump_mults), total=n_points, desc="Grid Search") if show_progress else enumerate(jump_mults)
        
        for i, jump_mult in grid_iterator:
            pvalues = np.zeros(samples_per_point)
            sample_iterator = tqdm(range(samples_per_point), leave=False, desc=f"Jump mult {jump_mult:.3f}") if show_progress else range(samples_per_point)
            
            for j in sample_iterator:
                pvalues[j] = run(jump_mult)
            
            mean_pvalues[i] = np.mean(pvalues)
            print(f"Jump mult {jump_mult:.3f}: mean p-value = {mean_pvalues[i]:.4f}")
            print()
            
        
        # Store results for this iteration
        all_jump_mults.append(jump_mults)
        all_mean_pvalues.append(mean_pvalues)
        
        # Update best overall result
        iter_best_idx = np.argmax(mean_pvalues)
        if mean_pvalues[iter_best_idx] > best_pvalue:
            best_pvalue = mean_pvalues[iter_best_idx]
            best_jump_mult = jump_mults[iter_best_idx]
        
        # if plot:
        #     # Plot current iteration
        #     plt.figure(figsize=(10, 6))
        #     plt.errorbar(jump_mults, mean_pvalues, fmt='o-', capsize=5, 
        #                 label=f'Iteration {iteration + 1}')
        #     plt.xscale('log')
        #     plt.xlabel('Jump Multiplier')
        #     plt.ylabel('Mean p-value')
        #     plt.title(f'Adaptive Grid Search - Iteration {iteration + 1}')
        #     plt.grid(True)
        #     plt.legend()
        #     plt.show()
        
        if iteration < n_refinements:
            # Calculate new search range around best point
            range_width = (current_max - current_min) * refinement_factor
            current_min = max(min_jump_mult, best_jump_mult - range_width/2)
            current_max = min(max_jump_mult, best_jump_mult + range_width/2)
    print()
    print("\nFinal Results:")
    print(f"Best jump multiplier: {best_jump_mult:.4f}")
    print(f"Best mean p-value: {best_pvalue:.4f}")
    
    if plot:
        # Plot all iterations together
        plt.figure(figsize=(12, 8))
        for i in range(len(all_jump_mults)):
            plt.errorbar(all_jump_mults[i], all_mean_pvalues[i], 
                        fmt='o-', capsize=5, label=f'Iteration {i+1}')
        plt.xscale('log')
        plt.xlabel('Jump Multiplier')
        plt.ylabel('Mean p-value')
        plt.title('Adaptive Grid Search - All Iterations')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{time_hor}_ML_VIZ_FIRST_RUN.svg', transparent=True)
        plt.show()
    
    return best_jump_mult, best_pvalue, all_jump_mults, all_mean_pvalues

# best_mult, best_pval, all_mults, all_means = adaptive_grid_search(time_hor=1000)