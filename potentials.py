import numpy as np


def d_poly__d_x(x, t, period=100):
    a1 = 0
    a2 = -3.2
    sin_scale = 3
    a3 = sin_scale*np.sin(2*np.pi*t/period)
    a4 = 0.1
    a5 = -(3/5) * a3
    a6 = 1

    all_scale = 1.13
    
    return all_scale*(a6*6*x**5 + a5*5*x**4 + a4*4*x**3 + a3*3*x**2 + a2*2*x + a1)

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