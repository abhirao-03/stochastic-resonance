# Stochastic Resonance in Climate Models.


Implementation of numerical methods for solving various types of Stochastic Differential Equations (SDEs), with a focus on climate modeling applications.

## Overview

This project provides a framework for solving different types of SDEs using various numerical schemes including:
- Euler-Maruyama method
- Milstein scheme
- Implicit methods (Full and Drift)
- Exact solutions for specific SDE types

## Features

### Supported SDE Models
- Langevin SDE
- Geometric Brownian Motion (GBM)
- Climate SDEs with varying potential functions.

### Numerical Methods
- **Euler-Maruyama**: First-order numerical scheme
- **Milstein**: Higher-order scheme with improved accuracy
- **Implicit Methods**: 
  - Full implicit scheme
  - Drift implicit scheme
- **Exact Solutions**: Available for Langevin and GBM models

### Key Components

## Usage

Example usage for solving a climate SDE with sinusoidal potential:

```python
import sde_models as sde
from dim_solvers import solver

# Create SDE instance
sin_sde = sde.climate_sde(
    x_init=0.0,
    epsilon=0.2,
    dt=0.1,
    time_horizon=1000,
    potential='sin'
)

# Initialize solver
sin_solver = solver(sin_sde)

# Solve using Euler-Maruyama method
sin_sim = sin_solver.euler_maruyama()

# Plot results
plt.plot(sin_sde.time_vec, sin_sim, 'o', label='sin')
plt.show()
```

## Implementation Details

### Model Parameters
- Time discretization with customizable step size
- Noise generation using JAX random number generation
- Support for multiple trajectories

### Solver Features
- Vectorized operations for improved performance
- Progress tracking during solving
- Support for both scalar and vector-valued SDEs

### Potential Functions
- Customizable potential functions
- Time-dependent potentials
- Support for instantaneous switching dynamics

## Mathematical Background

The project implements numerical solutions for SDEs of the form:

dX(t) = μ(X(t), t)dt + σ(X(t), t)dW(t)

where:
- $X(t)$ is the state variable
- $\mu(X(t), t)$ is the drift coefficient
- $\sigma(X(t), t)$ is the diffusion coefficient
- $W(t)$ is a Wiener process