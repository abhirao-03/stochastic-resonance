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

def sin_potential(x, t, period = 100):
    a = 1.1
    b = -3.4
    c = -1.7

    stationary = a*x**4 + b*x**2
    osc = jnp.sin((2*jnp.pi*t)/period)*x

    return stationary + osc + c

def final_potential(x, t, period=1000):
    a0 = 0
    a1 = 0
    a2 = -3.2
    a3 = 2*jnp.sin(2*jnp.pi*t/period)
    a4 = 0.1
    a5 = -(3/5) * a3
    a6 = 1

    return a6*x**6 + a5*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0

def d_final__d_x(x, t, period=100):
    sin_term = jnp.sin((2*jnp.pi*t)/period)
    return 6*x**5 -6*sin_term*x**4 + (4/10)*x**3 + 6*sin_term*x**2 - 6.4*x