import solvers as solvers
import sde_model as sde
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

black_sholes_SDE = sde.black_scholes_SDE(0.0001, 0.0002)
solver = solvers.solver(black_sholes_SDE)

implicit = solver.full_implicit()
euler = solver.euler_maruyama()
milstein = solver.milstein()
#exact = solver.exact_solution()            #currently not implemented

plt.plot(black_sholes_SDE.time_vec, implicit, label = 'implicit')
plt.plot(black_sholes_SDE.time_vec, euler, label = 'euler')
plt.plot(black_sholes_SDE.time_vec, milstein, label = 'milstein')
#plt.plot(black_sholes_SDE.time_vec, exact)
plt.xlabel('time')
plt.ylabel('$X(t)$')
plt.title('Different Methods')
plt.legend()
plt.show()