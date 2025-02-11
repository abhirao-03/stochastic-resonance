import numpy as np
import numpy.random as random
from optimiser import run


eps_init = 3.7
dt_init = 0.01
time_hor_init = 1000

eps_pert = 0.5
dt_pert = 0.9
time_hor_pert = 100

initial_simplex = np.array([[eps_init, dt_init, time_hor_init],
                            [eps_init + eps_pert, dt_init, time_hor_init],
                            [eps_init, dt_init*dt_pert, time_hor_init],
                            [eps_init, dt_init, time_hor_init + time_hor_pert]])

p_vals = np.empty(shape=(initial_simplex.shape[0], 2))
for idx, vertex in enumerate(initial_simplex):
    p_vals[idx] = np.array([run(vertex), vertex])


print()