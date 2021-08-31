# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:32:03 2021

@author: Mozaffarim
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

alpha=0.01
beta=0.1
gamma=0.01,
max_px_move=0.1,
convergence=0.01,
  
valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed',
                 'fixed-free', 'fixed-fixed', 'free-free']

x_axis = np.linspace(0, 1500, 1501, endpoint=True)
f = np.load("./x_test.npy")[20]  

snake_distants = 5

f = minmax_scale(f, feature_range=(0, 1))

# how many situations can be occured for five points

x = x_axis[::snake_distants]
  
n = len(x)
convergence_order = 10

xsave = np.empty((convergence_order, n))

# Build snake shape matrix for Euler equation
a = np.roll(np.eye(n), -1, axis=0) + \
    np.roll(np.eye(n), -1, axis=1) - \
    2*np.eye(n)  # second order derivative, central difference
b = np.roll(np.eye(n), -2, axis=0) + \
    np.roll(np.eye(n), -2, axis=1) - \
    4*np.roll(np.eye(n), -1, axis=0) - \
    4*np.roll(np.eye(n), -1, axis=1) + \
    6*np.eye(n)  # fourth order derivative, central difference
A = -alpha*a + beta*b

boundary_condition='fixed'
# Impose boundary conditions different from periodic:
sfixed = False
if boundary_condition.startswith('fixed'):
    A[0, :] = 0
    A[1, :] = 0
    A[1, :3] = [1, -2, 1]
    sfixed = True
efixed = False
if boundary_condition.endswith('fixed'):
    A[-1, :] = 0
    A[-2, :] = 0
    A[-2, -3:] = [1, -2, 1]
    efixed = True
sfree = False
if boundary_condition.startswith('free'):
    A[0, :] = 0
    A[0, :3] = [1, -2, 1]
    A[1, :] = 0
    A[1, :4] = [-1, 3, -3, 1]
    sfree = True
efree = False
if boundary_condition.endswith('free'):
    A[-1, :] = 0
    A[-1, -3:] = [1, -2, 1]
    A[-2, :] = 0
    A[-2, -4:] = [-1, 3, -3, 1]
    efree = True

# Only one inversion is needed for implicit spline energy minimization:
inv = np.linalg.inv(A + gamma*np.eye(n))

plt.figure(dpi=500)
plt.plot(x_axis, f, lw=0.3)
# plt.scatter(x, f[np.int64(x)], 8, color='green', marker='x')

# Explicit time stepping for image energy minimization:
for i in range(25000):
    
    fx = np.gradient(f[np.int64(x)], 1) + np.gradient(f[np.int64(x)], 2)
 
    if sfixed:
        fx[0] = 0
    if efixed:
        fx[-1] = 0
    if sfree:
        fx[0] *= 2
    if efree:
        fx[-1] *= 2
    xn = inv @ (gamma*x + fx)

    # Movements are capped to max_px_move per iteration:
    dx = max_px_move*np.tanh(xn-x)
   
    if sfixed:
        dx[0] = 0
    if efixed:
        dx[-1] = 0
    x += dx

    # Convergence criteria needs to compare to a number of previous
    # configurations since oscillations can occur.
    j = i % (convergence_order+1)
    if j < convergence_order:
        xsave[j, :] = x
    else:
        dist = np.min(np.max(np.abs(xsave-x[None, :]), 1))
        if dist < convergence:
            break

plt.scatter(x, f[np.int64(x)], 8, color='red', marker='x')


fnew = interp1d(np.int64(x), f[np.int64(x)], 'cubic')

plt.plot(x, fnew(x))