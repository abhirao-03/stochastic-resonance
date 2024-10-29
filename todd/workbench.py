import sde_models as sde
from dim_solvers import *
import matplotlib.pyplot as plt

non_osc_sde = sde.climate_sde_2()
em_solver_2 = solver(non_osc_sde)

em_sim_2 = em_solver_2.euler_maruyama()

plt.plot(non_osc_sde.time_vec, em_sim_2)
plt.title('Climate SDE E-M Scheme')
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.show()