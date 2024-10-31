import sde_models as sde
from dim_solvers import *
import matplotlib.pyplot as plt
import jax.numpy as jnp

sin_sde = sde.climate_sde(x_init = 0.0, epsilon=0.2, potential='sin')
pol_sde = sde.climate_sde(x_init = 0.0, epsilon=0.2)

sin_solver = solver(sin_sde)
pol_solver = solver(pol_sde)

sin_sim = sin_solver.euler_maruyama()
print('completed sin')
pol_sim = pol_solver.euler_maruyama()
print('completed polynomial')


plt.plot(sin_sde.time_vec, sin_sim, 'o', label='sin')
plt.plot(pol_sde.time_vec, pol_sim, 'x', label='polynomial')
plt.plot()
plt.title('Climate SDE E-M Scheme')
plt.ylim((-2.0, 2.0))
plt.legend()
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.show()