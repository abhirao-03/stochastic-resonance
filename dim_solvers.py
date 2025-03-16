import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from jax import grad, vmap


class solver():
    def __init__(self, sde):
        self.sde = sde

    def euler_maruyama(self):
        x = np.zeros((self.sde.num_steps, self.sde.num_trajectories))
        x[0] = self.sde.x_init

        for i in tqdm(range(self.sde.num_steps - 1)):
            curr_t = self.sde.time_vec[i]
            curr_x = x[i, :]
            dW = self.sde.noise[i, :]

            x[i+1, :] = curr_x + self.sde.mu(curr_x, curr_t)*self.sde.dt + self.sde.sigma(curr_x, curr_t) * dW

            # if np.isnan(x[i+1, :]).any() or np.isinf(x[i+1, :]).any():
            #     raise ValueError(f'Encountered: {x[i+1, :]}')

        return x
    
    def milstein(self):
        x = jnp.zeros((self.sde.num_steps, self.sde.num_trajectories))
        x = x.at[0, :].set(self.sde.x_init)

        for i in range(self.sde.num_steps - 1):
            t = self.sde.time_vec[i]
            dW = self.sde.noise[i + 1, :] - self.sde.noise[i, :]
            x = x.at[i+1, :].set(x[i, :] \
                                + vmap(self.sde.mu, in_axes=(0, None))(x[i, :], t) * self.sde.dt \
                                + vmap(self.sde.sigma, in_axes=(0, None))(x[i, :], t) * dW \
                                + vmap(self.sde.sigma, in_axes=(0, None))(x[i, :], t) * vmap(lambda z: grad(self.sde.sigma, argnums=(0))(z, t), in_axes=(0))(x[i, :]) * (dW**2 - self.sde.dt)
                                )
        return x