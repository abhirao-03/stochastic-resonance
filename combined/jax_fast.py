import jax.random as random
from jax import grad, vmap, jit
import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt


def sin_potential(x, t, period = 10000):
    a = 1.1
    b = -3.4
    c = -1.7

    stationary = a*x**4 + b*x**2
    osc = jnp.sin((2*jnp.pi*t)/period)*x

    return stationary + osc + c

def mu(x, t):
    return -grad(sin_potential, argnums=(0))(x, t)

def sigma(x=0.0, t=0.0):
    return 1.0


epsilon = 0.2
x_init = 1.0
dt = 0.1
time_horizon = 1000.0
num_steps = int(time_horizon/dt)
time = jnp.linspace(0, time_horizon, num_steps)
num_trajectories = 1
key = random.PRNGKey(1)
noise = jnp.sqrt(epsilon*dt) * random.normal(key, shape=(num_steps, num_trajectories))


def euler_maruyama(time):
    x = jnp.zeros((num_steps, num_trajectories))
    x = x.at[0, :].set(x_init)

    for i in range(num_steps - 1):
        t = time[i]
        dW = noise[i+1, :] - noise[i, :]
        x = x.at[i + 1, :].set(x[i, :] \
                        + vmap(mu, in_axes=(0, None))(x[i, :], t) * dt \
                        + vmap(sigma, in_axes=(0, None))(x[i, :], t) * dW
                        )
    return x

em_sim = jit(euler_maruyama)(time)

plt.plot(time, em_sim)
plt.title('Climate SDE E-M Scheme')
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.show()