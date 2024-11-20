import sde_models as sde
import dim_solvers as solvers
import sys

sys.path.append('')

import matplotlib.pyplot as plt
import numpy as np

num_trajectories = 100
climate_sde = sde.climate_sde(x_init=0, epsilon=0.75, dt=0.01, time_horizon=1000, num_trajectories=num_trajectories)
solver = solvers.solver(climate_sde)
em_sim = solver.euler_maruyama()

plt.plot(climate_sde.time_vec, em_sim[:, 0])
plt.legend()
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.ylim((-2, 2))
plt.tight_layout()
plt.show()

np.save('em_sim.npy', em_sim)

print("Completed Simulation")