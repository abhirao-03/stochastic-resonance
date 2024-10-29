from jax import grad, vmap
import numpy as np
import sde_models as sde


class solver():
    def __init__(self, sde):
        self.sde = sde

    def euler_maruyama(self):
        x = np.zeros((self.sde.num_steps, self.sde.num_trajectories))
        x[0, :] = self.sde.x_init

        for i in range(self.sde.num_steps - 1):
            t = self.sde.time_vec[i]
            dW = self.sde.noise[i+1, :] - self.sde.noise[i, :]
            x[i + 1, :] = x[i, :] \
                         + vmap(self.sde.mu, in_axes=(0, None))(x[i, :], t) * self.sde.dt \
                         + vmap(self.sde.sigma, in_axes=(0, None))(x[i, :], t) * dW
            
            print(f'Running EM iter {i}')
        return x

    def milstein(self):
        x = np.zeros((self.sde.num_steps, self.sde.num_trajectories))
        x[0, :] = self.sde.x_init

        for i in range(self.sde.num_steps - 1):
            t = self.sde.time_vec[i]
            dW = self.sde.noise[i + 1, :] - self.sde.noise[i, :]
            x[i + 1, :] = x[i, :] \
                         + vmap(self.sde.mu, in_axes=(0, None))(x[i, :], t) * self.sde.dt \
                         + vmap(self.sde.sigma, in_axes=(0, None))(x[i, :], t) * dW \
                         + vmap(self.sde.sigma, in_axes=(0, None))(x[i, :], t) * vmap(lambda z: grad(self.sde.sigma, argnums=(0))(z, t), in_axes=(0))(x[i, :]) * (dW**2 - self.sde.dt)
        return x

    def langevin_exact(self):
        assert self.sde != sde.langevin_SDE, 'This SDE is not supported. Inputs are a Langevin SDE'

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

    def gbm_exact(self):
        assert self.sde != sde.gbm_SDE, 'This SDE is not supported. Inputs are a Geometric Brownian Motion SDE'
        x = np.zeros(self.sde.num_steps, self.sde.num_trajectories)
        x[0, :] = self.sde.x_init

        for i in range(self.sde.num_steps - 1):
            t = self.sde.time_vec[i]
            x[i+1, :] = self.sde.x_init * np.exp((self.sde.MU - (self.sde.SIGMA**2)/2)*t - self.sde.noise[i, :])


    def full_implicit(self):

        assert self.sde != sde.gbm_SDE, 'This SDE is currently not supported for implicit methods'

        x = np.zeros((self.sde.num_steps, self.sde.num_trajectories))
        x[0, :] = self.sde.x_init

        for i in range(self.sde.num_steps - 1):
            dW = self.sde.noise[i + 1, :] - self.sde.noise[i, :]
            update = x[i, :] / (1 - self.sde.MU * self.sde.dt - self.sde.SIGMA*dW)
            x[i+1, :] = update
    
        return x
    
    def drift_implicit(self):

        assert self.sde != sde.gbm_SDE, 'This SDE is currently not supported for implicit methods'

        x = np.zeros((self.sde.num_steps, self.sde.num_trajectories))
        x[0, :] = self.sde.x_init

        for i in range(self.sde.num_steps - 1):
            dW = self.sde.noise[i + 1, :] - self.sde.noise[i, :]
            x[i+1, :] = x[i, :]*(1+self.sde.SIGMA * dW)/(1-self.sde.MU * self.sde.dt)
    
        return x