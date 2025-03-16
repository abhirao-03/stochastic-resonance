import sde_models as sde
from dim_solvers import *
import matplotlib.pyplot as plt

#osc_sde = sde.climate_sde()
non_osc_sde = sde.climate_sde_2(T=100000)
#em_solver = solver(osc_sde)
em_solver_2 = solver(non_osc_sde)

#em_sim = em_solver.euler_maruyama()
em_sim_2 = em_solver_2.euler_maruyama()

plt.plot(non_osc_sde.time_vec, em_sim_2)
plt.title('Climate SDE E-M Scheme')
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.show()