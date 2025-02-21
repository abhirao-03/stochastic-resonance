import numpy as np

def d_poly__d_x(x, t, period=100):
    t = 0
    a1 = 0
    a2 = -3.2
    sin_scale = 3
    a3 = sin_scale*np.sin(2*np.pi*t/period)
    a4 = 0.1
    a5 = -(3/5) * a3
    a6 = 1

    all_scale = 1.3
    
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



def d_triple__d_x(x):
    """
    Calculates the derivative of ax^2(x-c)^2(x-b)^3(x-d)^3
    
    Parameters:
        x: point at which to evaluate derivative
        a, b, c, d: coefficients (default to 1)
    """

    a = 0.005
    b = -1.9343
    c = 2.14
    d = 4.3
    # Breaking down the function into parts for clarity
    # f(x) = ax^2 * (x-c)^2 * (x-b)^3 * (x-d)^3
    
    # Using the product rule repeatedly
    term1 = 2 * x                      # derivative of x^2
    term2 = (x - c) ** 2              # (x-c)^2
    term3 = (x - b) ** 3              # (x-b)^3
    term4 = (x - d) ** 3              # (x-d)^3
    
    term5 = x ** 2                     # x^2
    term6 = 2 * (x - c)               # derivative of (x-c)^2
    
    term7 = x ** 2                     # x^2
    term8 = (x - c) ** 2              # (x-c)^2
    term9 = 3 * (x - b) ** 2          # derivative of (x-b)^3
    
    term10 = x ** 2                    # x^2
    term11 = (x - c) ** 2             # (x-c)^2
    term12 = (x - b) ** 3             # (x-b)^3
    term13 = 3 * (x - d) ** 2         # derivative of (x-d)^3
    
    # Combining all terms using the product rule
    derivative = a * (
        term1 * term2 * term3 * term4 +    # derivative of x^2 term
        term5 * term6 * term3 * term4 +    # derivative of (x-c)^2 term
        term7 * term8 * term9 * term4 +    # derivative of (x-b)^3 term
        term10 * term11 * term12 * term13  # derivative of (x-d)^3 term
    )
    
    return derivative