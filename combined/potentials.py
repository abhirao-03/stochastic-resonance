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

def poly_potential(x, t, period=100):
    a0 = 0
    a1 = 0
    a2 = -3.2
    sin_scale = 3
    a3 = sin_scale*jnp.sin(2*jnp.pi*t/period)
    a4 = 0.1
    a5 = -(3/5) * a3
    a6 = 1

    all_scale = 1.13

    return all_scale*(a6*x**6 + a5*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0)

# def d_poly__d_x(x, t, period=100):
#     a1 = 0
#     a2 = -3.2
#     a3 = 2*jnp.sin(2*jnp.pi*t/period)
#     a4 = 0.1
#     a5 = -(3/5) * a3
#     a6 = 1

#     all_scale = 0.5
    
#     return all_scale*(a6*6*x**5 + a5*5*x**4 + a4*4*x**3 + a3*3*x**2 + a2*2*x + a1)

def inst_switch(x, t, period=1000):
    a0 = 0
    a1 = 0
    a2 = -3.2
    a3 = 0.25 if t <= period/2 else 0.75
    a4 = 0.1
    a5 = -(3/5) * a3
    a6 = 1

    all_scale = 0.8

    return all_scale*(a6*x**6 + a5*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0)

# def d_inst_switch__d_x(x, t, period=100):
#     a1 = 0
#     a2 = -3.2
#     a3 = 0.25 if t <= period/2 else 0.75
#     a4 = 0.1
#     a5 = -(3/5) * a3
#     a6 = 1

#     all_scale = 0.8
    
#     return all_scale*(a6*6*x**5 + a5*5*x**4 + a4*4*x**3 + a3*3*x**2 + a2*2*x + a1)