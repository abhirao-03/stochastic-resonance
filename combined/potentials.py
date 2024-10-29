from jax import grad, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt

def poor_potential(x):
    return (x**4)/4 - (850.5/3)*x**3 + (241091.72/2)*x**2 - 22778441.868*x +1613920200.24048

def stable_potential(x):
    a = 1.1
    b = 3.4
    c = -3.0

    X_temp = a*(x - 283.3)**4 + b*(x - 283.3)**2

    return X_temp + c

def sin_potential(x, t, period = 10000):
    a = 1.1
    b = -3.4
    c = -1.7

    stationary = a*x**4 + b*x**2
    osc = jnp.sin((2*jnp.pi*t)/period)*x

    return stationary + osc + c

