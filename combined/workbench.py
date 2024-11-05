import sde_models as sde
from dim_solvers import *
import matplotlib.pyplot as plt
import jax.numpy as jnp

sin_sde = sde.climate_sde(x_init = 0.0, epsilon=0.2, dt=0.1, time_horizon=1000, potential='sin')
pol_sde = sde.climate_sde(x_init = 0.0, epsilon=2.0, dt=0.01, time_horizon=200, potential='polynomial')
inst_sde = sde.climate_sde(x_init = 0.0, epsilon=2.0, dt=0.1, time_horizon=1000, potential='inst_switch')


sin_solver = solver(sin_sde)
pol_solver = solver(pol_sde)
inst_solver = solver(inst_sde)

sin_sim = sin_solver.euler_maruyama()
print('completed sin')

pol_sim = pol_solver.euler_maruyama()
print('completed polynomial')

#inst_sim = inst_solver.euler_maruyama()
#print('completed instant')


plt.plot(sin_sde.time_vec, sin_sim, 'o', label='sin')
plt.plot(pol_sde.time_vec, pol_sim, 'x', label='polynomial')
# plt.plot(pol_sde.time_vec, inst_sim, 'x', label='polynomial')
plt.plot()
plt.title('Climate SDE E-M Scheme')
plt.ylim((-2.0, 2.0))
plt.legend()
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.show()