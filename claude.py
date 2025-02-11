import numpy as np
from scipy.stats import cramervonmises
from tqdm import tqdm
from numpy import random
import scipy.stats as stats


def const_neg_potential(x, t, period=100):
    """Defines the potential function."""
    t = 25
    a1, a2, a3, a4, a5, a6 = 0, -3.2, 3*np.sin(2*np.pi*t/period), 0.1, -(3/5)*3*np.sin(2*np.pi*t/period), 1
    all_scale = 1.13
    
    return all_scale * (a6*6*x**5 + a5*5*x**4 + a4*4*x**3 + a3*3*x**2 + a2*2*x + a1)

def mu(x, t):
    return -const_neg_potential(x, t)

def sigma(x, t, jump_mult, epsilon):
    return (epsilon * jump_mult) ** (1/2)


def simulate(x_init, dt, time_horizon, jump_mult, num_trajectories, delta=6):
    num_steps = int(time_horizon/dt)
    jump_times = np.empty((num_trajectories,))
    x = np.zeros((num_steps,))
    x[0] = x_init
    time_vec = np.linspace(0, time_horizon, num_steps)
    noise = random.normal(loc=0.0, scale=dt**(1/2), size=(num_trajectories, num_steps))

    global epsilon
    epsilon = (4.29 * 2 / np.log(time_horizon)) * jump_mult
    

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
                    sigma(curr_x, curr_t, jump_mult, epsilon) * dW)
            
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

def exp_cdf(x, rate):
    return 1 - np.exp(-rate * x)

class StochasticOptimizer:
    def __init__(self, perts, learning_rate=0.01, n_iterations=100, n_samples_per_iteration=5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_samples_per_iteration = n_samples_per_iteration
        self.perts = perts
        
    def objective(self, params):
        """
        Compute p-value for given parameters.
        params: [epsilon, dt, time_horizon]
        """
        jump_mult, dt, time_horizon = params

        # Run simulation
        x, jump_times = simulate(x_init, dt, time_horizon, jump_mult, num_trajectories)

        theoretical_rate = 1/(np.exp(1.17 * 2/epsilon))
        x_uniformed = exp_cdf(jump_times, theoretical_rate)
        
        return cramervonmises(x_uniformed, 'uniform').pvalue

    def estimate_gradient(self, params):
        """
        Estimate gradient using finite differences with multiple samples
        """
        gradients = []
        for _ in range(self.n_samples_per_iteration):
            grad = np.zeros_like(params)
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += self.perts[i]
                params_minus = params.copy()
                params_minus[i] -= self.perts[i]
                
                value_plus = self.objective(params_plus)
                value_minus = self.objective(params_minus)
                
                grad[i] = (value_plus - value_minus) / (2 * epsilon)
            gradients.append(grad)
            
        return np.mean(gradients, axis=0)
    
    def optimize(self, initial_params, param_bounds):
        """
        Run optimization with parameter bounds
        param_bounds: list of (min, max) tuples for each parameter
        """
        current_params = initial_params.copy()
        best_params = current_params.copy()
        best_value = float('-inf')
        history = []
        
        for i in tqdm(range(self.n_iterations)):
            # Estimate gradient
            gradient = self.estimate_gradient(current_params)
            
            # Update parameters
            current_params += self.learning_rate * gradient
            
            # Apply bounds
            for j, ((lower, upper), param) in enumerate(zip(param_bounds, current_params)):
                current_params[j] = np.clip(param, lower, upper)
            
            # Evaluate new parameters
            current_value = self.objective(current_params)
            history.append((current_params.copy(), current_value))
            
            # Update best parameters if necessary
            if current_value > best_value:
                best_value = current_value
                best_params = current_params.copy()
                
            print(f"Iteration {i+1}: P-value = {current_value:.4f}")
            print(f"Current params: epsilon={current_params[0]:.4f}, "
                  f"dt={current_params[1]:.4f}, time_horizon={current_params[2]:.4f}")
        
        return best_params, best_value, history

# Example usage

initial_params = np.array([3, 0.01, 1000])
param_bounds = [
    (1.0, 10.0),    # jump bounds
    (0.001, 0.1),   # dt bounds
    (100, 2000)     # time_horizon bounds
]

perts = np.array([0.2, 0.001, 100])

optimizer = StochasticOptimizer(
    learning_rate=0.01,
    n_iterations=50,
    n_samples_per_iteration=3,
    perts=perts
)

num_trajectories = 1000
x_init = -1

best_params, best_value, history = optimizer.optimize(initial_params, param_bounds)

print("\nOptimization Results:")
print(f"Best p-value: {best_value:.4f}")
print(f"Best parameters:")
print(f"epsilon: {best_params[0]:.4f}")
print(f"dt: {best_params[1]:.4f}")
print(f"time_horizon: {best_params[2]:.4f}")