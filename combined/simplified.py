import numpy as np
import matplotlib.pyplot as plt


def d_poly__d_x(x, t, period=100):
    a1 = 0
    a2 = -3.2
    sin_scale = 3
    a3 = sin_scale*np.sin(2*np.pi*t/period)
    a4 = 0.1
    a5 = -(3/5) * a3
    a6 = 1

    all_scale = 1.3
    
    return all_scale*(a6*6*x**5 + a5*5*x**4 + a4*4*x**3 + a3*3*x**2 + a2*2*x + a1)

def sin_potential_derivative(x, t, period=1000):
    a = 1.1
    b = -3.4

    # Derivative of the stationary term
    stationary_derivative = 4 * a * x**3 + 2 * b * x
    
    # Derivative of the oscillating term
    osc_derivative = np.sin((2 * np.pi * t) / period)
    
    # Combine the derivatives
    return stationary_derivative + osc_derivative


x_init = 0.0
epsilon = 0.4
dt = 0.1
time_horizon = 100
num_steps = int(time_horizon/dt)
noise = np.random.normal(loc=0.0, scale=epsilon*dt, size=(num_steps,))
x_poly = np.zeros(shape=(num_steps, ))
time_vec = np.linspace(0, time_horizon, num=num_steps + 1)
x_poly[0] = x_init



def mu(x, t):
    return -d_poly__d_x(x, t)

def sigma(_x, _t):
    return 1.0

def euler_maruyama():
    for i in range(num_steps - 1):
        t = time_vec[i]
        x = x_poly[i]

        dW = noise[i+1] - noise[i]

        x_poly[i+1] = x + mu(x, t) * dt + sigma(x, t) * dW


        print(f"On iteration {i}")
    return x_poly


x_poly = euler_maruyama()


plt.plot(time_vec[:-1], x_poly, label = 'polynomial]')
plt.legend()
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.ylim((-2, 2))
plt.tight_layout()
plt.show()
print()