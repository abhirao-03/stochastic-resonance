import numpy as np
from jax import grad, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt

def poor_potential(x):
    return (x**4)/4 - (850.5/3)*x**3 + (241091.72/2)*x**2 - 22778441.868*x +1613920200.24048

def stable_potential(x):
    a = 0.01
    b = 0

    X_temp = a*(x - 278.6)**2*(x - 288.6)**2+b

    return X_temp

def sin_potential(x, t):
    a = 1.1
    b = -3.4
    c = -1.7
    T = 100000

    stationary = a*x**4 + b*x**2
    osc = np.sin((2*np.pi*t)/T)*x

    return stationary + osc + c