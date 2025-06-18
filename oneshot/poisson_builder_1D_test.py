# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:28:09 2022

@author: GD5264
"""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import time
import matplotlib.pyplot as plt

from poisson_builder_1D import poisson_builder_1D

script_name = 'poisson_builder_1D_test'

plt.close('all')

Nx = 50
f = np.zeros(Nx) + 1
for ii in range(Nx):
    if (ii > Nx//2):
        f[ii] = 2
ismissing = np.zeros([Nx], bool)
ismissing[15:35] = True
f[ismissing] = 0.0

f = np.arange(Nx)
for ii in range(Nx):
    if (ii > Nx//2):
        f[ii] = f[Nx//2] + 3*(ii-Nx//2)
ismissing = np.zeros([Nx], bool)
ismissing[15:35] = True
f[ismissing] = 0.0

c0 = np.zeros(Nx)
c0[~ismissing] = 1

alpha = 0.1

c2_d_ref = np.zeros(Nx) + alpha
c2_d_ref[ismissing] = alpha

c2_n_ref = np.zeros(Nx) + alpha
c2_n_ref[ismissing] = alpha

A_d, b_d = poisson_builder_1D(c0, c2_d_ref, f, bc='Dirichlet')
A_d_sp = csc_matrix(A_d)
u_d_ref = spsolve(A_d_sp, b_d)

A_n, b_n = poisson_builder_1D(c0, c2_n_ref, f, bc='Neumann')
A_n_sp = csc_matrix(A_n)
u_n_ref = spsolve(A_n_sp, b_n)


err_ref = np.sum((u_n_ref[ismissing]-u_d_ref[ismissing])**2)
print(script_name + ': err_ref =', err_ref)

err = np.zeros(Nx) + np.nan
tic = time.time()
print(script_name + ': computing ...', flush=True)
for ii in range(Nx):

    if not(ismissing[ii]):
        continue

    c2_d = c2_d_ref.copy()
    c2_d[ii-1] *= 0.01
    c2_d[ii] *= 0.01
    c2_d[ii+1] *= 0.01
    A_d, b_d = poisson_builder_1D(c0, c2_d, f, bc='Dirichlet')
    A_d_sp = csc_matrix(A_d)
    u_d = spsolve(A_d_sp, b_d)

    c2_n = c2_n_ref.copy()
    c2_n[ii-1] *= 0.01
    c2_n[ii] *= 0.01
    c2_n[ii+1] *= 0.01
    A_n, b_n = poisson_builder_1D(c0, c2_n, f, bc='Neumann')
    A_n_sp = csc_matrix(A_n)
    u_n = spsolve(A_n_sp, b_n)

    err[ii] = np.sum((u_n[ismissing]-u_d[ismissing])**2)

    """
    plt.figure()
    plt.plot(u_d_ref, 'o-')
    plt.plot(u_n_ref, 'o-')
    plt.plot(u_d, 'o-')
    plt.plot(u_n, 'o-')
    plt.legend(['ud ref', 'un ref', 'ud', 'un'])

    if (ii == Nx//2):
        stop
    """

toc = time.time()
print(script_name + ': done in', toc-tic, 'sec', flush=True)

plt.figure()
plt.plot(err, 'o-')


err_thres = np.nanmin(err)

c2_d = c2_d_ref.copy()
c2_d[err<=err_thres] *= 0.01
c2_n = c2_n_ref.copy()
c2_n[err<=err_thres] *= 0.01
A_d, b_d = poisson_builder_1D(c0, c2_d, f, bc='Dirichlet')
A_d_sp = csc_matrix(A_d)
u_d = spsolve(A_d_sp, b_d)

plt.figure()
plt.plot(u_d, 'o-')
plt.plot(u_d_ref)
plt.legend(['topo', 'ref'])
