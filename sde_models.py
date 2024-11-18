import numpy as np
import numpy.random as random
from jax import vmap
from potentials import d_poly__d_x as polynomial_potential

class model_params():
    def __init__(self, x_init=1.0, dt=0.1, time_horizon=10.0, num_trajectories=1):
        self.x_init = x_init
        self.dt = dt
        self.time_horizon = time_horizon
        self.num_steps = int(self.time_horizon/self.dt)
        self.time_vec = np.linspace(0, self.time_horizon, self.num_steps)
        self.num_trajectories = num_trajectories
        
        self.noise = random.normal(loc=0.0, scale = dt**(1/2), size=(self.num_steps, self.num_trajectories))


class climate_sde(model_params):
    def __init__(self, x_init=0.0, epsilon=0.1, dt=0.1,  time_horizon=100, num_trajectories=1):
        super().__init__(x_init=x_init, dt=dt, time_horizon=time_horizon, num_trajectories=num_trajectories)
        self.epsilon = epsilon
        self.noise = random.normal(loc=0.0, scale = dt**(1/2), size=(self.num_steps, self.num_trajectories))

    def mu(self, x, t):
            return -polynomial_potential(x,t)
    
    def sigma(self, x=0.0, t=0.0):
        return self.epsilon ** (1/2)