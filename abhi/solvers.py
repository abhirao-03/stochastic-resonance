from jax import grad
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
            t = self.sde.time_vec[i]
            dW = self.sde.noise[i+1] - self.sde.noise[i]

            x[i+1] = x[i] \
                     + self.sde.mu(x[i], t) * self.sde.dt \
                     + self.sde.sigma(x[i], t) * dW
        
        return x

    def milstein(self):
        x = np.zeros(self.sde.num_steps)
        x[0] = self.sde.x_init

        for i in range(self.sde.num_steps - 1):
            t = self.sde.time_vec[i]
            dW = self.sde.noise[i + 1] - self.sde.noise[i]
            x[i + 1] = x[i]\
                       + self.sde.mu(x[i], t) * self.sde.dt \
                       + self.sde.sigma(x[i], t) * dW \
                       + self.sde.sigma(x[i], t) * grad(self.sde.sigma, argnums=(1))(x[i], t) * (dW**2 - self.sde.dt)
        return x

    def exact_solution(self):
        assert self.sde != sde.langevin_SDE, 'This SDE is currently not supported for exact solution'

        x = np.zeros(self.sde.num_steps)
        x[0] = self.sde.x_init

        # simple sum integral approximation
        integral_term = np.zeros(self.sde.num_steps)
        for i in range(self.sde.num_steps - 1):
            t = self.sde.time_vec[i]
            integral_term[i+1] = integral_term[i]+np.exp(self.sde.theta*t)*self.sde.noise[i]

        for i in range(self.sde.num_steps - 1):
            t = self.sde.time_vec[i]
            initial_term   = self.sde.x_init * np.exp(self.sde.theta * t)
            drift_term     = self.sde.MU * (1-np.exp(self.sde.theta * t))
            diffusion_term = self.sde.SIGMA * integral_term[i]

            x[i+1] = initial_term + drift_term + diffusion_term
        
        return x

    def full_implicit(self):

        assert self.sde != sde.black_sholes_SDE, 'This SDE is currently not supported for implicit methods'

        x = np.zeros(self.sde.num_steps)
        x[0] = self.sde.x_init

        for i in range(self.sde.num_steps - 1):
            delta_W = self.sde.noise[i + 1] - self.sde.noise[i]
            update = x[i] / (1 - self.sde.MU * self.sde.dt - self.sde.SIGMA*delta_W)
            x[i+1] = update
    
        return x