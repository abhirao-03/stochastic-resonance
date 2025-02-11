import numpy as np
import numpy.random as random
from optimiser import run

x_init = -1
dt = 0.01
time_horizon = 1000
jump_mult = 3
num_trajectories = 1000

initial_simplex = np.array([
                            [x_init,        dt,         time_horizon,       jump_mult,     num_trajectories],
                            [x_init + 0.01, dt,         time_horizon,       jump_mult,     num_trajectories],
                            [x_init,        dt - 0.001, time_horizon,       jump_mult,     num_trajectories],
                            [x_init,        dt,         time_horizon + 100, jump_mult,     num_trajectories],
                            [x_init,        dt,         time_horizon,       jump_mult + 1, num_trajectories],
                            [x_init,        dt,         time_horizon,       jump_mult,     num_trajectories - 10]
                            ])

p_vals = np.empty(shape=(initial_simplex.shape[0]))

for idx, vertex in enumerate(initial_simplex):
    p_vals[idx] = run(vertex)

idx = np.argsort(p_vals)
idx = idx[::-1]
sorted_p_vals = p_vals[idx]
sorted_simplex = initial_simplex[idx]

def calculate_centroid(simplex):
    centroid = np.mean(simplex[:-1], axis=0)
    return centroid

def reflect(centroid, worst_point, alpha=1.0):
    reflection = centroid + alpha*(centroid - worst_point)
    return reflection

def expand(centroid, reflected_point, gamma=2.0):
    expansion = centroid + gamma*(reflected_point - centroid)
    return expansion

def contract_outside(centroid, reflected_point, beta=0.5):
    cont_out = centroid + beta*(reflected_point - centroid)
    return cont_out

def contract_inside(centroid, worst_point, beta=0.5):
    cont_in = centroid + beta*(worst_point - centroid)
    return cont_in

def shrink(simplex, best_vertex, sigma=0.5):
    new_simplex = np.empty(simplex.shape)
    new_simplex[0] = best_vertex

    new_vertex = best_vertex + sigma*(simplex[1:] - best_vertex)
    new_simplex[1:] = new_vertex
    
    return new_simplex
    
print()