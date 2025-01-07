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

def const_neg_potential(x, t, period=100):
    t = 25                                   # re-define t to place minima on negative.
    a1 = 0
    a2 = -3.2
    sin_scale = 3
    a3 = sin_scale*np.sin(2*np.pi*t/period)
    a4 = 0.1
    a5 = -(3/5) * a3
    a6 = 1

    all_scale = 1.13
    
    return all_scale*(a6*6*x**5 + a5*5*x**4 + a4*4*x**3 + a3*3*x**2 + a2*2*x + a1)

def const_pos_potential(x, t, period=100):
    t = 75                                    # re-define t to place minima on positive.
    a1 = 0
    a2 = -3.2
    sin_scale = 3
    a3 = sin_scale*np.sin(2*np.pi*t/period)
    a4 = 0.1
    a5 = -(3/5) * a3
    a6 = 1

    all_scale = 1.13
    
    return all_scale*(a6*6*x**5 + a5*5*x**4 + a4*4*x**3 + a3*3*x**2 + a2*2*x + a1)

def d_V_pot(x, t, min_val=5):
    if x < -1:
        return -min_val
    if -1 < x and x < 0:
        return min_val
    if 0 < x and x < 1:
        return -min_val
    if 1 < x:
        return min_val
    if x == -1 or x == 1 or x == 0:
        return 0
    
def d_V_pot(x, t, min_val=5):
    # Multi-dimensional V potential.
    x = np.array(x)
    y = np.zeros_like(x)
    y[x < -1] = -min_val
    y[np.logical_and(-1 < x, x < 0)] = min_val
    y[np.logical_and(0 < x, x < 1)] = -min_val
    y[x > 1] = min_val
    y[np.logical_or(x == -1, x == 1, x == 0)] = 0
    return y