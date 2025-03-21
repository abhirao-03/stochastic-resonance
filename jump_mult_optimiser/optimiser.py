import numpy as np
from numpy import random
from tqdm import tqdm
import scipy.stats as stats


shallow = False

dt = 0.01
time_horizon = 100
num_trajectories = 1000
num_steps = int(time_horizon/dt)

well_depth = 1.17 if shallow else 4.29
x_init = 1 if shallow else -1
epsilon = ((well_depth * 2)/np.log(time_horizon))
time_vec = np.linspace(0, time_horizon, num_steps)


def const_neg_potential(x, t, period=100):
    """Defines the potential function."""
    t = 25
    a1 = 0
    a2 = -3.2
    a3 = 3 * np.sin(2 * np.pi * t / period)
    a6 = 1
    a4 = (-3 / 2) * a6 + (8 / 5)
    a5 = -(3 / 5) * a3
    all_scale = 1.3

    # Clip x to prevent overflow
    x_clipped = np.clip(x, -1e2, 1e2)  # Adjust bounds as needed

    return all_scale * (a6 * 6 * x_clipped**5 + a5 * 5 * x_clipped**4 + 
                        a4 * 4 * x_clipped**3 + a3 * 3 * x_clipped**2 + 
                        a2 * 2 * x_clipped + a1)

def mu(x, t):
    return -const_neg_potential(x, t)

def sigma(x, t, epsilon):
    return (epsilon) ** (1/2)

def simulate(jump_mult, noise, jump_threshold=1, delta=10):
    jump_times = np.empty((num_trajectories,))
    trajectories = np.zeros((num_trajectories, num_steps))
    
    for j in tqdm(range(num_trajectories)):
        x = np.zeros((num_steps,))
        x[0] = x_init
        steps_above_threshold = 0
        jump_confirmed = False
        first_cross_index = None
        
        for i in range(num_steps - 1):
            curr_x = x[i]
            dW = noise[j, i]
            
            # Update trajectory
            x[i+1] = curr_x + mu(curr_x, time_vec[i]) * dt + sigma(curr_x, time_vec[i], epsilon*jump_mult) * dW
            
            # Check if above jump_threshold
            if x[i+1] >= jump_threshold:
                steps_above_threshold += 1
                if steps_above_threshold == 1:
                    first_cross_index = i + 1  # Index where threshold was first crossed
            else:
                steps_above_threshold = 0
                first_cross_index = None
            
            # Check if persistence criteria met
            if steps_above_threshold >= delta:
                # Backtrack to find last crossing from <0 to >=0 before first_cross_index
                last_zero_cross = None
                if first_cross_index is not None:
                    for k in range(first_cross_index, 0, -1):
                        if x[k] >= 0 and x[k-1] < 0:
                            last_zero_cross = time_vec[k]
                            break
                jump_times[j] = last_zero_cross if last_zero_cross is not None else np.nan
                jump_confirmed = True
                break
        
        if not jump_confirmed:
            jump_times[j] = np.nan
        
        trajectories[j] = x
    
    return trajectories, jump_times


def exp_cdf(x, jump_mult):
    theoretical_rate = 1/(np.exp(well_depth * 2/(epsilon * jump_mult)))
    return 1 - np.exp(-theoretical_rate * x)

def run(jump_mult):
    noise = random.normal(loc=0.0, scale=dt**(1/2), size=(num_trajectories, num_steps))

    _, jump_times = simulate(jump_mult, noise)

    valid_jump_times = jump_times.copy()
    valid_jump_times[np.isnan(valid_jump_times)] = time_horizon

    x_transformed = exp_cdf(valid_jump_times, jump_mult)

    met = stats.cramervonmises(x_transformed, 'uniform').pvalue

    return met