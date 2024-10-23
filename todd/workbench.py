import solvers as solvers
import sde_model as sde
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

def verify_noise_uniqueness(sde):
    noise = sde.noise
    noise_tuples = [tuple(noise[:, trial]) for trial in range(sde.num_trials)]
    unique_noise = len(set(noise_tuples))
    if unique_noise == sde.num_trials:
        print("All trials have unique noise sequences.")
    else:
        print(f"{sde.num_trials - unique_noise} trials share identical noise sequences.")

trials = 3

black_sholes_SDE = sde.black_scholes_SDE(0.001, 0.002, num_trials=trials)
black_sholes_SDE.num_trials = trials
verify_noise_uniqueness(black_sholes_SDE)
assert black_sholes_SDE.num_trials == trials
solver = solvers.solver(black_sholes_SDE)

euler    = solver.euler_maruyama()
milstein = solver.milstein()
full_implicit = solver.full_implicit()
drift_implicit = solver.drift_implicit()
#exact    = solver.exact_solution()

plt.plot(black_sholes_SDE.time_vec, euler[:, 0], linestyle= "-", color="r", label='Euler Maruyama')
plt.plot(black_sholes_SDE.time_vec, euler[:, 0], linestyle= "-", color="r", label='Exact')
#plt.plot(black_sholes_SDE.time_vec, euler[:, 1], linestyle= "-.", color="g", label='Trial 2')
#plt.plot(black_sholes_SDE.time_vec, euler[:, 2], linestyle= "--", color="b", label='Trial 3')

# for trial in range(trials): 
    # plt.plot(black_sholes_SDE.time_vec, euler[:, trial], linestyle= "--", color="b", label=f'euler method trial{trial}')
    # plt.plot(black_sholes_SDE.time_vec, milstein[:, trial],linestyle="-.", color="g")
    # plt.plot(black_sholes_SDE.time_vec, full_implicit[:, trial],linestyle=":", color="r")
    # plt.plot(black_sholes_SDE.time_vec, drift_implicit[:, trial],linestyle="-", color="m")
    # plt.plot(black_sholes_SDE.time_vec, exact[:, trial], linestyle="-", color="red")

plt.title('Euler-Maruyama & Milstein Simulation (Multiple Trials)')
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.legend()
plt.show()
