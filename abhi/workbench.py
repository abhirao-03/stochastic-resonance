import solvers as solvers
import sde_model as sde
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

langevin = sde.langevin_SDE()

solver = solvers.solver(langevin)

euler_maruyama = solver.euler_maruyama()


print()