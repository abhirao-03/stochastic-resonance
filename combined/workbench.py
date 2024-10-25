import sde_models as sde
from dim_solvers import *
import matplotlib.pyplot as plt

osc_sde = sde.climate_sde()
em_solver = solver(osc_sde)

em_sim = em_solver.euler_maruyama()

plt.plot(osc_sde.time_vec, em_sim)
plt.title('Climate SDE E-M Scheme')
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.show()