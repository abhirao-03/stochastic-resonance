import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

class model_params():
    def __init__(self, x_init=1.0, dt=0.01, time_horizon=10.0):
        self.x_init = x_init
        self.dt = dt
        self.t_end = time_horizon
        self.num_steps = int(self.t_end/self.dt)
        self.time_vec = np.linspace(0, self.t_end, self.num_steps)
        self.noise = random.normal(loc=0.0, scale=np.sqrt(self.dt), size=self.num_steps)

class langevin_SDE(model_params):
    def __init__(self, mean=0.0, std=0.1, tau=0.05):
        #distribution parameters
        self.MU = mean
        self.SIGMA = std
        self.tau = tau
        self.theta = 1/self.tau

    def mu(self, x, _t):
            return self.theta * (self.MU - x)

    def sigma(self, _y, _t):
            return self.SIGMA * np.sqrt(2/self.tau)

class black_scholes_SDE(model_params):
    def __init__(self, mu, sigma, theta = 1.0):
        super().__init__()
        self.MU = mu
        self.SIGMA = sigma
        self.theta = theta

    def mu(self, x, _t):
        return self.MU * x
    
    def sigma(self, x, _t):
        return self.SIGMA * x


print()