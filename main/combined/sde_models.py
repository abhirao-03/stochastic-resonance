import numpy.random as random
import numpy as np
from jax import grad
from potentials import *

class model_params():
    def __init__(self, x_init=1.0, dt=0.01, time_horizon=10.0, num_trajectories=1):
        self.x_init = x_init
        self.dt = dt
        self.t_end = time_horizon
        self.num_steps = int(self.t_end/self.dt)
        self.time_vec = np.linspace(0, self.t_end, self.num_steps)
        self.num_trajectories = num_trajectories
        
        random.seed(1)
        self.noise = random.normal(loc=0.0, scale=self.dt, size=(self.num_steps, self.num_trajectories))


class langevin_SDE(model_params):
    def __init__(self, mean=0.0, std=0.1, tau=0.05, num_trajectories=1):
        #distribution parameters
        super().__init__(num_trajectories=num_trajectories)
        self.MU = mean
        self.SIGMA = std
        self.tau = tau
        self.theta = 1/self.tau

    def mu(self, x, _t):
            return self.theta * (self.MU - x)

    def sigma(self, _y, _t):
            return self.SIGMA * np.sqrt(2/self.tau)


class gbm_SDE(model_params):
    def __init__(self, mu, sigma, theta = 1.0, num_trajectories=1):
        super().__init__(num_trajectories=num_trajectories)
        self.MU = mu
        self.SIGMA = sigma
        self.theta = theta

    def mu(self, x, _t):
        return self.MU * x
    
    def sigma(self, x, _t):
        return self.SIGMA * x


class climate_sde(model_params):
    def __init__(self, epsilon=0.2):
        super().__init__(x_init=1.0, dt = 0.1, time_horizon=1000.0, num_trajectories=1)
        self.epsilon = epsilon
        self.noise = random.normal(loc=0.0, scale=epsilon*self.dt, size=(self.num_steps, self.num_trajectories))

    def mu(self, x, t):
        return -grad(sin_potential, argnums=(0))(x, t)
    
    def sigma(self, x, t):
        return 1.0
    
class climate_sde_2(model_params):
    def __init__(self, T=100000, epsilon=0.2):
        self.time_horizon = 1000
        super().__init__(x_init=288.6, dt = 0.1, time_horizon=self.time_horizon, num_trajectories=1)
        self.epsilon = epsilon
        self.T = T
        self.noise = random.normal(loc=0.0, scale=epsilon*self.dt, size=(self.num_steps, self.num_trajectories))

    def mu(self, x, t):
        Q=0.1
        return -grad(stable_potential, argnums=(0))(x)-Q*(np.sin((2*np.pi*t)/self.T))
    
    def sigma(self, x, t):
        return 1