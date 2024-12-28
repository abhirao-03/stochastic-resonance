import sde_models as sde
import dim_solvers as solvers
from potentials import d_poly__d_x, d_V_pot, const_neg_potential, const_pos_potential

import matplotlib.pyplot as plt
import numpy as np


num_trajectories = 1
climate_sde = sde.climate_sde(x_init=0,
                              epsilon=0.75,
                              dt=0.01,
                              time_horizon=1000,
                              num_trajectories=num_trajectories,
                              potential=)
solver = solvers.solver(climate_sde)

print("STARTED SIMULATION")
em_sim = solver.euler_maruyama()
print("COMPLETED SIMULATION")

print("PLOTTING FIRST TRAJECTORY")

plt.plot(climate_sde.time_vec, em_sim[:, 0])
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.title('First Trajectory')
plt.ylim((-2, 2))
plt.tight_layout()
plt.show()

print("SAVED SIMULATED VALUES TO 'em_sim.npy'")