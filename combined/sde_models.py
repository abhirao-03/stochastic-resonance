import jax.random as random
import jax.numpy as jnp
from jax import grad
from potentials import *

class model_params():
    def __init__(self, x_init=1.0, dt=0.1, time_horizon=10.0, num_trajectories=1):
        self.x_init = x_init
        self.dt = dt
        self.t_end = time_horizon
        self.num_steps = int(self.t_end/self.dt)
        self.time_vec = jnp.linspace(0, self.t_end, self.num_steps)
        self.num_trajectories = num_trajectories
        
        self.key = random.PRNGKey(1)
        self.noise = jnp.sqrt(self.dt) * random.normal(self.key, shape=(self.num_steps, self.num_trajectories))


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
            return self.SIGMA * jnp.sqrt(2/self.tau)


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
    def __init__(self, x_init=0.0, epsilon=0.1, dt=0.1,  time_horizon=1000, potential='sin'):
        super().__init__(x_init=x_init, dt = dt, time_horizon=time_horizon, num_trajectories=1)
        self.potential = potential
        self.epsilon = epsilon
        self.noise = jnp.sqrt(epsilon * self.dt) * random.normal(self.key, shape=(self.num_steps, self.num_trajectories))

    def mu(self, x, t):
        if self.potential == 'sin':
            return -grad(sin_potential, argnums=(0))(x, t)

        elif self.potential == 'inst_switch':
            return -d_inst_switch__d_x(x, t)
        
        elif self.potential == 'polynomial':
            return -d_poly__d_x(x, t)
    
    def sigma(self, x=0.0, t=0.0):
        return 1.0