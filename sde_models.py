import numpy as np
import numpy.random as random

class model_params():
    def __init__(self, x_init=1.0, dt=0.1, time_horizon=10.0, num_trajectories=1):
        self.x_init = np.ones((1, num_trajectories)) * x_init
        self.dt = dt
        self.time_horizon = time_horizon
        self.num_steps = int(self.time_horizon/self.dt)
        self.time_vec = np.linspace(0, self.time_horizon, self.num_steps)
        self.num_trajectories = num_trajectories
        
        print('GENERATING NOISE')
        self.noise = random.normal(loc=0.0, scale = dt**(1/2), size=(self.num_steps, self.num_trajectories))
        print('COMPLETED NOISE GENERATION')

class climate_sde(model_params):
    def __init__(self, potential, x_init=0.0, dt=0.1,  time_horizon=100, num_trajectories=1, want_jumps: bool = True, jump_mult: float = 1.5):
        super().__init__(x_init=x_init, dt=dt, time_horizon=time_horizon, num_trajectories=num_trajectories)
        self.jump_mult = jump_mult
        
        if want_jumps == True:
            self.epsilon = (4.29 / np.log(self.time_horizon)) * self.jump_mult
        else:
            self.epsilon = 4.29 / np.log(self.time_horizon) - 0.001
        
        self.potential = potential

    def mu(self, x, t):
        return -self.potential(x, t)
    
    def sigma(self, x=0.0, t=0.0):
        return self.epsilon ** (1/2)

class langevin_SDE(model_params):
    def __init__(self, mean=0.0, std=0.1, tau=0.05, num_trajectories=1):
        #distribution parameters
        super().__init__(num_trajectories=num_trajectories)
        self.MU = mean
        self.SIGMA = std
        self.tau = tau
        self.theta = 1/self.tau

    def mu(self, x, _t):
            return self.theta * (self.MU - x)

    def sigma(self, _y, _t):
            return self.SIGMA * np.sqrt(2/self.tau)


class gbm_SDE(model_params):
    def __init__(self, mu, sigma, theta = 1.0, num_trajectories=1):
        super().__init__(num_trajectories=num_trajectories)
        self.MU = mu
        self.SIGMA = sigma
        self.theta = theta

    def mu(self, x, _t):
        return self.MU * x
    
    def sigma(self, x, _t):
        return self.SIGMA * x