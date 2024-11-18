import numpy as np

class solver():
    def __init__(self, sde):
        self.sde = sde

    def euler_maruyama(self):
        x = np.zeros((self.sde.num_steps, self.sde.num_trajectories))
        x[0] = self.sde.x_init

        for i in range(self.sde.num_steps - 1):
            curr_t = self.sde.time_vec[i]
            curr_x = x[i]
            dW = self.sde.noise[i]

            x[i+1, :] = curr_x + self.sde.mu(x[i, :], curr_t)*self.sde.dt + self.sde.sigma(x[i, :], curr_t) * dW
        return x