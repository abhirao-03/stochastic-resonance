import numpy as np
from DT_optimiser import run
import matplotlib.pyplot as plt
from tqdm import tqdm

def adaptive_dt_grid_search(
    n_points: int = 6,
    min_dt: float = 0.0001,
    max_dt: float = 0.01,
    samples_per_point: int = 5,
    n_refinements: int = 3,
    refinement_factor: float = 0.2,
    show_progress: bool = True,
    plot: bool = True):
    """
    Perform adaptive grid search over dt parameter with progressive refinement.
    Starts with larger dt values first to detect any potential errors.
    
    Args:
        n_points: Number of grid points to evaluate in each iteration
        min_dt: Initial minimum dt value
        max_dt: Initial maximum dt value
        samples_per_point: Number of evaluations per grid point
        n_refinements: Number of refinement iterations
        refinement_factor: Factor to determine width of refined search region
        show_progress: Whether to show progress bars
        plot: Whether to show plots
        
    Returns:
        best_dt: Overall best dt value found
        best_pvalue: Best mean p-value found
        all_dts: List of arrays of dt values for each iteration
        all_mean_pvalues: List of arrays of mean p-values for each iteration
    """

    all_dts = []
    all_mean_pvalues = []
    
    current_min = min_dt
    current_max = max_dt
    best_dt = None
    best_pvalue = -np.inf
    
    for iteration in range(n_refinements + 1):
        print(f"\nIteration {iteration + 1}/{n_refinements + 1}")
        print(f"Searching range: [{current_min:.6f}, {current_max:.6f}]")
        print()
        
        # Create grid points with geometric spacing
        dt_values = np.geomspace(current_min, current_max, n_points)
        
        # Sort in descending order to start with larger values first
        dt_values = np.sort(dt_values)[::-1]
        
        mean_pvalues = np.zeros(n_points)
        
        # Evaluate each grid point
        grid_iterator = tqdm(enumerate(dt_values), total=n_points, desc="Grid Search") if show_progress else enumerate(dt_values)
        
        for i, dt in grid_iterator:
            pvalues = np.zeros(samples_per_point)
            sample_iterator = tqdm(range(samples_per_point), leave=False, desc=f"dt {dt:.6f}") if show_progress else range(samples_per_point)
            
            try:
                for j in sample_iterator:
                    pvalues[j] = run(dt)  # Using original run function with dt as parameter
                
                mean_pvalues[i] = np.mean(pvalues)
                print(f"dt {dt:.6f}: mean p-value = {mean_pvalues[i]:.4f}")
                print()
            
            except Exception as e:
                print(f"Error with dt={dt:.6f}: {str(e)}")
                print("Skipping this value and continuing with smaller dt values")
                mean_pvalues[i] = -np.inf  # Ensure this value is not selected as optimal
        
        # Clean up any error values before storing/plotting
        valid_indices = mean_pvalues > -np.inf
        clean_dt_values = dt_values[valid_indices]
        clean_mean_pvalues = mean_pvalues[valid_indices]
        
        if len(clean_dt_values) == 0:
            print("No valid dt values found in this iteration. Terminating search.")
            break
        
        # Store results for this iteration (only valid values)
        all_dts.append(clean_dt_values)
        all_mean_pvalues.append(clean_mean_pvalues)
        
        # Update best overall result
        iter_best_idx = np.argmax(clean_mean_pvalues)
        if clean_mean_pvalues[iter_best_idx] > best_pvalue:
            best_pvalue = clean_mean_pvalues[iter_best_idx]
            best_dt = clean_dt_values[iter_best_idx]
        
        if plot and len(clean_dt_values) > 0:
            # Plot current iteration
            plt.figure(figsize=(10, 6))
            plt.errorbar(clean_dt_values, clean_mean_pvalues, fmt='o-', capsize=5, 
                        label=f'Iteration {iteration + 1}')
            plt.xscale('log')
            plt.xlabel('dt (Time Step)')
            plt.ylabel('Mean p-value')
            plt.title(f'Adaptive Grid Search for dt - Iteration {iteration + 1}')
            plt.grid(True)
            plt.legend()
            plt.show()
        
        if iteration < n_refinements:
            # Calculate new search range around best point
            range_width = (current_max - current_min) * refinement_factor
            current_min = max(min_dt, best_dt - range_width/2)
            current_max = min(max_dt, best_dt + range_width/2)
    
    if best_dt is not None:
        print()
        print("\nFinal Results:")
        print(f"Best dt value: {best_dt:.6f}")
        print(f"Best mean p-value: {best_pvalue:.4f}")
        
        if plot:
            # Plot all iterations together
            plt.figure(figsize=(12, 8))
            for i in range(len(all_dts)):
                plt.errorbar(all_dts[i], all_mean_pvalues[i], 
                            fmt='o-', capsize=5, label=f'Iteration {i+1}')
            plt.xscale('log')
            plt.xlabel('dt (Time Step)')
            plt.ylabel('Mean p-value')
            plt.title('Adaptive Grid Search for dt - All Iterations')
            plt.grid(True)
            plt.legend()
            plt.show()
    else:
        print("No valid dt values were found during the search.")
    
    return best_dt, best_pvalue, all_dts, all_mean_pvalues

# Run the adaptive grid search
if __name__ == "__main__":
    best_dt, best_pval, all_dts, all_means = adaptive_dt_grid_search()