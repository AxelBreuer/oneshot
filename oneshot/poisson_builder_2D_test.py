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

from poisson_builder_2D import poisson_builder_2D

script_name = 'poisson_builder_2D_test'

plt.close('all')

Nx = 50
Ny = 50
f = np.zeros([Nx, Ny])
for ii in range(Nx):
    for jj in range(Ny):
        if (((ii-25)**2+(jj-25)**2) < 10**2):
            f[ii, jj] = 1
ismissing = np.zeros([Nx, Ny], bool)
ismissing[25:45, 25:45] = True
f[ismissing] = 0.0

c0 = np.zeros([Nx, Ny])
c0[~ismissing] = 1

alpha = 0.01

c2_d_ref = np.zeros([Nx, Ny]) + alpha
c2_d_ref[ismissing] = alpha

c2_n_ref = np.zeros([Nx, Ny]) + alpha
c2_n_ref[ismissing] = alpha

A_d, b_d = poisson_builder_2D(c0, c2_d_ref, f, bc='Dirichlet')
A_d_sp = csc_matrix(A_d)
u_d_1d = spsolve(A_d_sp, b_d)
# reshape 1d vector to 2d array
u_d = np.reshape(u_d_1d, [Nx, Ny], 'F')

A_n, b_n = poisson_builder_2D(c0, c2_n_ref, f, bc='Neumann')
A_n_sp = csc_matrix(A_n)
u_n_1d = spsolve(A_n_sp, b_n)
# reshape 1d vector to 2d array
u_n = np.reshape(u_n_1d, [Nx, Ny], 'F')

err_ref = np.sum((u_n[ismissing]-u_d[ismissing])**2)
print(script_name + ': err_ref =', err_ref)

err = np.zeros([Nx, Ny]) + np.nan
tic = time.time()
print(script_name + ': computing ...', flush=True)
for ii in range(Nx):
    for jj in range(Ny):

        if not(ismissing[ii, jj]):
            continue

        c2_d = c2_d_ref.copy()
        c2_d[ii, jj] = c2_d[ii, jj]*0.01
        A_d, b_d = poisson_builder_2D(c0, c2_d, f, bc='Dirichlet')
        A_d_sp = csc_matrix(A_d)
        u_d_1d = spsolve(A_d_sp, b_d)
        # reshape 1d vector to 2d array
        u_d = np.reshape(u_d_1d, [Nx, Ny], 'F')

        c2_n = c2_n_ref.copy()
        c2_n[ii, jj] = c2_n[ii, jj]*0.01
        A_n, b_n = poisson_builder_2D(c0, c2_n, f, bc='Neumann')
        A_n_sp = csc_matrix(A_n)
        u_n_1d = spsolve(A_n_sp, b_n)
        # reshape 1d vector to 2d array
        u_n = np.reshape(u_n_1d, [Nx, Ny], 'F')

        err[ii, jj] = np.sum((u_n[ismissing]-u_d[ismissing])**2)

toc = time.time()
print(script_name + ': done in', toc-tic, 'sec', flush=True)

plt.figure()
plt.pcolor(err)
plt.colorbar()
plt.title('|u_d - u_n|')

err_thres = 1.025e-4

plt.figure()
plt.pcolor(err < err_thres)
plt.colorbar()
plt.title('|u_d - u_n| < ' + str(err_thres) + '(err without perturbation = '+ str(err_ref)+')')

plt.figure()
plt.pcolor(f + (err < err_thres))
plt.colorbar()
plt.title('ref picture and thresholded perturbation')

c2_d = c2_d_ref.copy()
c2_d[err<err_thres] = 0.01 * c2_d[err < err_thres]
c2_n = c2_n_ref.copy()
c2_n[err<err_thres] = 0.01 * c2_n[err < err_thres]
A_d, b_d = poisson_builder_2D(c0, c2_d, f, bc='Dirichlet')
A_d_sp = csc_matrix(A_d)
u_d_1d = spsolve(A_d_sp, b_d)
# reshape 1d vector to 2d array
u_d = np.reshape(u_d_1d, [Nx, Ny], 'F')

plt.figure()
plt.pcolor(u_d)
plt.title('inpainted')
