import numpy as np
from tqdm import tqdm

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

            if np.isnan(x[i+1, :]).any() or np.isinf(x[i+1, :]).any():
                raise ValueError(f'Encountered: {x[i+1, :]}')

        return x