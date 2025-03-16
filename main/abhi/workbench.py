import solvers as solvers
import sde_model as sde
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

black_sholes_SDE = sde.black_scholes_SDE(0.0001, 0.0002)
solver = solvers.solver(black_sholes_SDE)

euler    = solver.euler_maruyama()
milstein = solver.milstein()
#exact = solver.exact_solution()            #currently not implemented

full_implicit  = solver.full_implicit()
drift_implicit = solver.drift_implicit()

plt.plot(black_sholes_SDE.time_vec, euler, label = 'euler', linestyle=':')
plt.plot(black_sholes_SDE.time_vec, milstein, label = 'milstein', linestyle='-')
plt.plot(black_sholes_SDE.time_vec, full_implicit, label = 'implicit', linestyle='--')
plt.plot(black_sholes_SDE.time_vec, drift_implicit,label = 'em implicit', linestyle='-.')
#plt.plot(black_sholes_SDE.time_vec, exact)
plt.xlabel('time')
plt.ylabel('$X(t)$')
plt.title('Different Methods')
plt.legend()
plt.show()