import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

class model():
    def __init__(self, mean=0.0, std=0.1, tau=0.05, x_init=0.0, dt=0.01, time_horizon=10.0):
        #distribution parameters
        self.MU = mean
        self.SIGMA = std
        
        #time parameters
        self.x_init = x_init
        self.dt = dt
        self.tau = tau
        self.theta = 1/self.tau
        self.t_end = time_horizon
        self.num_steps = int(self.t_end/self.dt)
        self.time_vec = np.linspace(0, self.t_end, self.num_steps)
        self.noise = random.normal(loc=0.0, scale=np.sqrt(self.dt), size=self.num_steps)

