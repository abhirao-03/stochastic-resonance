import sde_model as sde
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

class solver():
    def __init__(self, sde):
        self.sde = sde

    def euler_maruyama(self):
        x = np.zeros(self.sde.num_steps)
        x[0] = self.sde.x_init
        
        for i in range(self.sde.num_steps - 1):
            t = self.sdetime_vec[i]
            x[i+1] = x[i] \
                     + self.sde.mu(x[i], t) * self.sde.dt \
                     + self.sde.sigma(x[i], t) * self.sde.noise[i]
        
        return x

    def milstein(self):
        x = np.zeros(self.sde.num_steps)
        x[0] = self.sde.x_init

        for i in range(self.sde.num_steps - 1):
            t = self.sde.time_vec[i]
            milstein = ((self.sde.sigma(x[i], t))/2) * x[i] * (self.sde.noise[i]**2 - self.sde.dt)

            x[i+1] = x[i]\
                     + self.sde.mu(t, x[i]) * self.sde.dt\
                     + self.sde.sigma(t, x[i]) * self.sde.noise[i] \
                     + milstein
        
        return x

    def exact_solution(self):
        x = np.zeros(self.sde.num_steps)
        x[0] = self.sde.x_init

        # simple sum integral approximation
        integral_term = np.zeros(self.sde.num_steps)
        for i in range(self.sde.num_steps - 1):
            t = self.sde.time_vec[i]
            integral_term[i+1] = integral_term[i]+np.exp(self.sde.theta*t)*self.sde.noise[i]

        for i in range(self.sde.num_steps - 1):
            t = self.sde.time_vec[i]
            initial_term   = self.sde.x_init * np.exp(self.theta * t)
            drift_term     = self.sde.MU * (1-np.exp(self.theta * t))
            diffusion_term = self.sde.SIGMA * integral_term[i]

            x[i+1] = initial_term + drift_term + diffusion_term
        
        return x