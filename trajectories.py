import sde_models as sde
from dim_solvers import solver
from potentials import d_poly__d_x, d_V_pot, const_neg_potential, const_pos_potential
import json

with open('sim_settings.json') as f:
    settings = json.load(f)

import matplotlib.pyplot as plt
import numpy as np

num_trajectories = 5
climate_sde = sde.climate_sde(x_init = 0,
                              epsilon = 0.75,
                              dt = 0.01,
                              time_horizon = 1000,
                              num_trajectories = num_trajectories,
                              potential = d_poly__d_x)

em_solver = solver(climate_sde)

print("STARTED SIMULATION")
em_sim = em_solver.euler_maruyama()
print("COMPLETED SIMULATION")

print("PLOTTING FIRST TRAJECTORY")

plt.plot(climate_sde.time_vec, em_sim[:, 0])
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.title('First Trajectory')
plt.ylim((-2, 2))
plt.tight_layout()
plt.show()

if settings['save_trajectories'] == True:
    np.save('em_sim.npy', em_sim)

print("SAVED SIMULATED VALUES TO 'em_sim.npy'")