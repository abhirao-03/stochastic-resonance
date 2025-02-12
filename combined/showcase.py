from sde import gbm_SDE
from dim_solvers import solver

my_sde = gbm_SDE(mu=0.1, sigma=0.1)     # Define the SDE model
my_solver = solver(my_sde)              # Define the solver

x_em = my_solver.euler_maruyama()       # Solve using EM method
x_mil = my_solver.milstein()            # Solve using Milstein method
x_exact = my_solver.gbm_exact()         # The exact GBM solution.