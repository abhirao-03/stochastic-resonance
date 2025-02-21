import numpy as np
from params import model_params

class climate_sde(model_params):
    def __init__(self,
                 potential,
                 x_init=0.0,
                 dt=0.1,
                 time_horizon=100,
                 num_trajectories=1,
                 want_jumps: bool = True,
                 jump_mult: float = 1.5):
        
        super().__init__(x_init=x_init,
                         dt=dt,
                         time_horizon=time_horizon,
                         num_trajectories=num_trajectories)
        
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
    