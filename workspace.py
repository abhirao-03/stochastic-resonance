import sde_models as sde
import dim_solvers as solvers

import matplotlib.pyplot as plt
import numpy as np

num_trajectories = 1000
climate_sde = sde.climate_sde(x_init=0, epsilon=0.75, dt=0.01, time_horizon=1000, num_trajectories=num_trajectories)
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

np.save('em_sim.npy', em_sim)
print("SAVED SIMULATED VALUES TO 'em_sim.npy'")