import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
from optimiser import run

def grid_search_jump_mult(
    n_points: int = 15,
    min_jump_mult: float = 3.02,
    max_jump_mult: float = 3.12,
    samples_per_point: int = 1,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform grid search over jump time multiplier to maximize Cramer-von Mises p-value.
    
    Args:
        n_points: Number of grid points to evaluate
        min_jump_mult: Minimum jump multiplier to consider
        max_jump_mult: Maximum jump multiplier to consider
        samples_per_point: Number of evaluations per grid point to handle stochasticity
        show_progress: Whether to show progress bars
        
    Returns:
        best_jump_mult: Jump multiplier that produced highest mean p-value
        best_pvalue: Highest mean p-value found
        jump_mults: Array of all jump multipliers evaluated
        mean_pvalues: Array of mean p-values for each jump multiplier
        std_pvalues: Array of standard deviations of p-values
    """
    # Create grid points with geometric spacing
    jump_mults = np.geomspace(min_jump_mult, max_jump_mult, n_points)
    all_pvalues = np.zeros(n_points)
    
    # Outer progress bar for grid points
    grid_iterator = tqdm(enumerate(jump_mults), total=n_points, desc="Grid Search")
    
    # Evaluate each grid point multiple times
    for i, jump_mult in grid_iterator:
        pvalues = np.zeros(samples_per_point)
        
        # Inner progress bar for samples at each point
        sample_iterator = tqdm(range(samples_per_point), leave=False, desc=f"Jump mult {jump_mult:.3f}")
        
        for j in sample_iterator:
            pvalues[j] = run(jump_mult)
        
        all_pvalues[i] = np.mean(pvalues)
        
        print(f"Jump mult {jump_mult:.3f}: mean p-value = {all_pvalues[i]:.4f}")
    
    # Find best result
    best_idx = np.argmax(all_pvalues)
    best_jump_mult = jump_mults[best_idx]
    best_pvalue = all_pvalues[best_idx]
    
    # Check if best value is at boundary
    if best_idx in [0, len(jump_mults)-1]:
        print("\nWarning: Best value found at grid boundary. Consider extending search range.")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(jump_mults, all_pvalues, fmt='o-', capsize=5)
    plt.xscale('log')
    plt.xlabel('Jump Multiplier')
    plt.ylabel('Mean p-value')
    plt.title('Grid Search Results with Error Bars')
    plt.grid(True)
    plt.show()
    
    return best_jump_mult, best_pvalue, jump_mults, all_pvalues

best_mult, best_pval, mults, means = grid_search_jump_mult()