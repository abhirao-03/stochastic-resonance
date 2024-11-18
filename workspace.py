import numpy as np
import dim_solvers as solvers
import sde_models as sde

import matplotlib.pyplot as plt

climate_sde = sde.climate_sde(x_init=0.0, epsilon=0.75, dt=0.01, time_horizon=100, num_trajectories=1)
solver = solvers.solver(climate_sde)


trajectories = 100
em_tracked = np.zeros(shape=(trajectories, climate_sde.num_steps))


for i in range(trajectories):
    print(f'Simulating trajectory {i}')

    climate_sde = sde.climate_sde(x_init=0.0, epsilon=0.75, dt=0.01, time_horizon=100, num_trajectories=1)

    solver = solvers.solver(climate_sde)
    
    em_sim = solver.euler_maruyama()
    
    em_tracked[i, :] = em_sim.reshape((climate_sde.num_steps, ))


plt.plot(climate_sde.time_vec, em_sim.reshape((climate_sde.num_steps,)), label = 'polynomial')
plt.legend()
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.ylim((-2, 2))
plt.tight_layout()
plt.show()

np.save('em_tracked.npy', em_tracked)
print()