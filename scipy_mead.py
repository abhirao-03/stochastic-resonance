from scipy.optimize import minimize
from optimiser import run
import numpy as np

dt = 0.01
time_hor = 1000
jump_mult = 3

bounds = np.array([(1e-10,0.1), (1,100000), (0,100000)])

x_init = np.array([dt, time_hor, jump_mult])



res = minimize(run, x_init, bounds=bounds)

print()