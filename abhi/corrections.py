import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.random as random

class model():
    def __init__(self, mean=10.0, std=1.0, x_init=5.0, dt=0.001, time_horizon=1.0, tau=0.05):
        #distribution parameters
        self.mean = mean
        self.std = std

        #time parameters
        self.x_init = x_init
        self.dt = dt
        self.tau = tau
        self.t_end = time_horizon
        self.num_steps = int(self.t_end/self.dt)
        self.time_vec = jnp.linspace(0, self.t_end, self.num_steps)
        self.noise = random.normal(loc=0.0, shape=(self.num_steps,))

    def euler__maruyama_langevin(self):
        x = jnp.zeros(self.num_steps)
        x[0] = self.x_init

        # Compute outside to avoid repeated computation as stated in ref[2]
        sigma_func = self.std * jnp.sqrt(2. / self.tau)

        for i in range(self.num_steps - 1):
            Z = self.noise[i]

            mu_func = -(x[i] - self.mean) / self.tau

            x[i+1] = x[i] \
                     + mu_func * self.dt \
                     + sigma_func * Z
        
        return x

    def milstein_langevin(self):
        x = jnp.zeros(self.num_steps)
        x[0] = self.x_init

        sigma_func = self.std * jnp.sqrt(2. / self.tau)

        for i in range(self.num_steps - 1):
            Z = self.noise[i]

            mu_func = -(x[i] - self.mean) / self.tau
            milstein_term = (1/2) * (sigma_func) * (0) * Z

            x[i+1] = x[i] \
                    + mu_func * self.dt \
                    + sigma_func * Z  \
                    + milstein_term

        return x