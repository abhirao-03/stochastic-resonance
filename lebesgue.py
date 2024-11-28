import numpy as np
import pandas as pd
from tqdm import tqdm

data = np.load('./results/em_sim.npy')
data = data.T

time_vec = np.linspace(0, 1000, int((1000)/0.01))

all_interval_lengths = []
all_interval_values  = []
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
        else:
            all_interval_lengths.append(interval)
            all_interval_values.append('negative')

data_builder = {'trajectory': trajectory_number,
                'interval_length': all_interval_lengths,
                'interval_value': all_interval_values}

df = pd.DataFrame(data=data_builder)
df.to_pickle('results/intervals.pkl')