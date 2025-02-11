
import numpy as np
from numpy import random
from tqdm import tqdm
import scipy.stats as stats

def run(vertex:np.array):
    
    # dt = 0.01
    # time_horizon = 1000

    dt = vertex[0]
    time_horizon = vertex[1]

    num_steps = int(time_horizon/dt)

    num_trajectories = 1000

    x_init = -1
    jump_mult = vertex[2]
    epsilon = (4.29 * 2 / np.log(time_horizon)) * jump_mult

    t = np.linspace(0, time_horizon, num_steps)
    
    def const_neg_potential(x, t, period=100):
        """Defines the potential function."""
        t = 25
        a1, a2, a3, a4, a5, a6 = 0, -3.2, 3*np.sin(2*np.pi*t/period), 0.1, -(3/5)*3*np.sin(2*np.pi*t/period), 1
        all_scale = 1.13
        
        return all_scale * (a6*6*x**5 + a5*5*x**4 + a4*4*x**3 + a3*3*x**2 + a2*2*x + a1)

    def mu(x, t):
        return -const_neg_potential(x, t)

    def sigma(x, t):
        return epsilon ** (1/2)


    def simulate(x_init, dt, num_trajectories, num_steps, time_vec, delta=6):
        jump_times = np.empty((num_trajectories,))
        x = np.zeros((num_steps,))
        x[0] = x_init
        
        for j in tqdm(range(num_trajectories)):
            sign_change_detected = False
            steps_since_change = 0
            initial_sign = np.sign(x[0])
            potential_jump_time = None
            
            for i in range(num_steps - 1):
                curr_t = time_vec[i]
                curr_x = x[i]
                dW = random.normal(loc=0.0, scale=dt**(1/2))
                
                x[i+1] = (curr_x + 
                        mu(curr_x, curr_t) * dt + 
                        sigma(curr_x, curr_t) * dW)
                
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

    print('simulating')
    _, jump_times = simulate(x_init, dt, num_trajectories, num_steps, t)

    theoretical_rate = 1/(np.exp(1.17 * 2/epsilon))

    def exp_cdf(x, rate=theoretical_rate):
        return 1 - np.exp(-rate * x)

    x_uniformed = exp_cdf(jump_times)

    pvalue = stats.cramervonmises(x_uniformed, 'uniform').pvalue
    print()
    print(f'p_value: {pvalue}')
    print(f'dt: {dt}')
    print(f'time horizon: {time_horizon}')
    print(f'jump_mult: {jump_mult}')
    print()
    return 1 - pvalue