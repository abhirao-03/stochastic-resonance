import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd


time_horizon = 1000.0
dt = 0.1
num_steps = int(time_horizon/dt)

t = jnp.linspace(0, time_horizon, num_steps)
x = jnp.load('em_sim.npy')
x = x.reshape((num_steps,))

plt.plot(t[x > 0], x[x > 0], 'o', label = 'positive well')
plt.plot(t[x < 0], x[x < 0], 'x', label = 'negative well')
plt.legend()
plt.show()

data = {
        'time'  : t,
        'x_val' : x
        }

pd.DataFrame(data)