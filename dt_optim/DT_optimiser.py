import numpy as np
from numpy import random
from tqdm import tqdm
import scipy.stats as stats

x_init = -1
dt_min = 0.001
time_horizon = 1000
num_trajectories = 1000
jump_mult = 3
epsilon = ((4.29 * 2)/np.log(time_horizon)) * jump_mult
max_num_steps = int(time_horizon/dt_min)

print('generating noise')
max_noise = random.normal(loc=0.0, scale=dt_min**(1/2), size=(num_trajectories, max_num_steps))
print('noise generated')

def const_neg_potential(x, t, period=100):
        """Defines the potential function."""
        t = 25
        a1 = 0
        a2 = -3.2
        a3 = 3*np.sin(2*np.pi*t/period)
        a6 = 1
        a4 = (-3/2)*a6 + (8/5)
        a5 = -(3/5)*a3
        all_scale = 1.3
        
        return all_scale * (a6*6*x**5 + a5*5*x**4 + a4*4*x**3 + a3*3*x**2 + a2*2*x + a1)

def mu(x, t):
    return -const_neg_potential(x, t)

def sigma(x, t):
    return (epsilon) ** (1/2)


def simulate(dt: float, delta=6):

    step_mult = int(dt / dt_min)

    num_steps = int(time_horizon/dt)
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
            base_idx = i * step_mult

            aggregated_noise = np.sum(max_noise[j, base_idx : base_idx+step_mult]) / np.sqrt(step_mult)
            
            x[i+1] = (curr_x + 
                    mu(curr_x, curr_t) * dt + 
                    sigma(curr_x, curr_t) * aggregated_noise)
            
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

def exp_cdf(x):
    theoretical_rate = 1/(np.exp(1.17 * 2/(epsilon)))
    return 1 - np.exp(-theoretical_rate * x)

def run(dt):
    _, jump_times = simulate(dt)

    x_transformed = exp_cdf(jump_times)

    met = stats.cramervonmises(x_transformed, 'uniform').pvalue

    return met



met = run(0.01)

print()