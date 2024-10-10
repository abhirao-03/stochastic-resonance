import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

class model():
    def __init__(self, mean=0.0, std=0.1, tau=0.05, x_init=0.0, dt=0.001, time_horizon=1.0):
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

    def mu(self, x, _t):
            return self.theta * (self.MU - x)

    def sigma(self, _y, _t):
            return self.sigma * np.sqrt(2/self.tau)

    def euler_maruyama(self):
        x = np.zeros(self.num_steps)
        x[0] = self.x_init
        
        for i in range(self.num_steps - 1):
            t = self.time_vec[i]
            x[i+1] = x[i] \
                     + self.mu(x[i], t) * self.dt \
                     + self.sigma(x[i], t) * self.noise[i]
        
        return x, self.time_vec

    def milstein(self):
        x = np.zeros(self.num_steps)
        x[0] = self.x_init

        for i in range(self.num_steps - 1):
            t = self.time_vec[i]
            milstein = ((self.sigma(x[i], t)**2)/2) * x[i] * (self.noise[i]**2 - self.dt)

            x[i+1] = x[i]\
                     + self.mu(t, x[i]) * self.dt\
                     + self.sigma(t, x[i]) * self.noise[i] \
                     + milstein
        
        return x, self.time_vec

    def exact_solution(self):
        x = np.zeros(self.num_steps)
        x[0] = self.x_init

        # simple sum integral approximation
        integral_term = np.zeros(self.num_steps)
        for i in range(self.num_steps - 1):
            t = self.time_vec[i]
            integral_term[i+1] = integral_term[i]+np.exp(self.theta*t)*self.noise[i]

        for i in range(self.num_steps):
            t = self.time_vec[i]
            initial_term   = self.x_init * np.exp(self.theta * t)
            drift_term     = self.MU * (1-np.exp(self.theta * t))
            diffusion_term = self.SIGMA * integral_term[i]

            x[i+1] = initial_term + drift_term + diffusion_term
        
        return x

print()