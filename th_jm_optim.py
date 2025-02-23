import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from tqdm import tqdm
import scipy.stats as stats


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


results = []
time_horizons = [100,500,1000,5000]

for th in time_horizons:

    x_init = -1
    dt = 0.01
    time_horizon = th
    num_trajectories = 1000
    epsilon = ((4.29 * 2)/np.log(time_horizon))
    num_steps = int(time_horizon/dt)
    noise = random.normal(loc=0.0, scale=dt**(1/2), size=(num_trajectories, num_steps))

    def const_neg_potential(x, t, period=100):
            """Defines the potential function."""
            t = 25
            a1, a2, a3, a4, a5, a6 = 0, -3.2, 3*np.sin(2*np.pi*t/period), 0.1, -(3/5)*3*np.sin(2*np.pi*t/period), 1
            all_scale = 1.3
            
            return all_scale * (a6*6*x**5 + a5*5*x**4 + a4*4*x**3 + a3*3*x**2 + a2*2*x + a1)

    def mu(x, t):
        return -const_neg_potential(x, t)

    def sigma(x, t, epsilon):
        return (epsilon) ** (1/2)

    def simulate(jump_mult: int, delta=6):
        jump_times = np.empty((num_trajectories,))
        x = np.zeros((num_steps,))
        x[0] = x_init
        time_vec = np.linspace(0, time_horizon, num_steps)
        
        for j in tqdm(range(num_trajectories)):
            sign_change_detected = False
            steps_since_change = 0
            initial_sign = np.sign(x[0])
            potential_jump_time = None
            
            for i in range(num_steps - 1):
                curr_t = time_vec[i]
                curr_x = x[i]
                dW = noise[j, i]
                
                x[i+1] = (curr_x + 
                        mu(curr_x, curr_t) * dt + 
                        sigma(curr_x, curr_t, epsilon * jump_mult) * dW)
                
                # If we haven't detected a sign change yet
                if not sign_change_detected:
                    if np.sign(x[i+1]) != initial_sign:
                        sign_change_detected = True
                        steps_since_change = 0
                        potential_jump_time = curr_t
                
                # If we're monitoring a potential zone transition
                elif steps_since_change < delta:
                    steps_since_change += 1
                    # If sign reverts back to initial sign, cancel the monitoring
                    if np.sign(x[i+1]) == initial_sign:
                        sign_change_detected = False
                        steps_since_change = 0
                        potential_jump_time = None
                    
                    # If we've monitored for delta steps and sign change persists
                    if steps_since_change == delta:
                        jump_times[j] = potential_jump_time
                        break
            
            # If we reach the end without confirming a transition
            if not sign_change_detected or steps_since_change < delta:
                jump_times[j] = np.nan
        
        return x, jump_times

    def exp_cdf(x, jump_mult):
        theoretical_rate = 1/(np.exp(1.17 * 2/(epsilon * jump_mult)))
        return 1 - np.exp(-theoretical_rate * x)

    def run(jump_mult):
        _, jump_times = simulate(jump_mult)

        x_transformed = exp_cdf(jump_times, jump_mult)

        met = stats.cramervonmises(x_transformed, 'uniform').pvalue

        return met

    best_jump_mult, best_p_value, _, _ = adaptive_grid_search(time_horizon)

    results.append((time_horizon, best_jump_mult, best_p_value))


print(results)