import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

import sde_models as sde
import dim_solvers as solvers
from potentials import d_poly__d_x, d_V_pot, const_neg_potential, const_pos_potential

# import json
with open('sim_settings.json') as f:
    settings = json.load(f)

num_trajectories = 100
climate_sde = sde.climate_sde(x_init = 0,
                              want_jumps=True,
                              jump_mult=settings['jump_mult'],
                              dt = 0.01,
                              time_horizon = 100,
                              num_trajectories = num_trajectories,
                              potential = const_neg_potential)
solver = solvers.solver(climate_sde)

print("STARTED SIMULATION")
em_sim = solver.euler_maruyama()
print("COMPLETED SIMULATION")

print("PLOTTING FIRST TRAJECTORY")

plt.plot(climate_sde.time_vec, em_sim[:, 0], alpha=0.1)
plt.plot(climate_sde.time_vec, em_sim[:, 1], alpha=0.1)
plt.plot(climate_sde.time_vec, em_sim[:, 2], alpha=0.1)
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.title('First Trajectory')
plt.ylim((-2, 2))
plt.tight_layout()
plt.show()

print("STARTING WELL ANALYSIS")

data = em_sim
np.save('results/em_sim.npy', em_sim)
data = data.T

time_vec = np.linspace(0, 1000, int((1000)/0.01))

all_interval_lengths = []
all_interval_values  = []
all_well_types       = []
trajectory_number    = []


for i in tqdm(range(len(data))):
    d1 = data[i]
    neg_well = d1 < 0
    arg_tracker = [0] + [j  for j in range(1, len(neg_well)) if neg_well[j] != neg_well[j - 1]] + [len(time_vec) - 1]

    for k in range(1, len(arg_tracker)):
        trajectory_number.append(i)
        prev_t = time_vec[arg_tracker[k - 1]]
        curr_t = time_vec[arg_tracker[k]]

        interval = (curr_t - prev_t)

        if d1[arg_tracker[k - 1]] >= 0:
            all_interval_lengths.append(interval)
            all_interval_values.append('positive')
            all_well_types.append('shallow well')
        else:
            all_interval_lengths.append(interval)
            all_interval_values.append('negative')
            all_well_types.append('deep well')

data_builder = {'trajectory': trajectory_number,
                'interval_length': all_interval_lengths,
                'interval_value': all_interval_values,
                'well_type': all_well_types}

df = pd.DataFrame(data=data_builder)
df.to_pickle('results/test.pkl')