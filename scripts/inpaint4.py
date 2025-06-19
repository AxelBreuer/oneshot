# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:15:17 2021

@authors: Axel Breuer and Didier Auroux
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import time

from oneshot.poisson_builder_2D import poisson_builder_2D
# from oneshot.poisson_builder_2Db import poisson_builder_2Db as poisson_builder_2D
# from oneshot.poisson_builder_2Dc import poisson_builder_2Dc as poisson_builder_2D
from oneshot.poisson_verify import poisson_verify


"""
The purpose of this script is to implement the inpainting algorithm
described in "A one-shot inpainting algorithm based on the
topological asymptotic analysis" by Auroux and Masmoudi.
"""

# close all current plots
plt.close('all')

"""
build example to inpaint
"""


example = 'circle'

if (example == 'circle'):
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

elif (example == 'square'):
    Nx = 64
    Ny = 64
    f = np.zeros([Nx, Ny])
    f[:Nx//2, :Ny//2] = 1.0
    ismissing = np.zeros([Nx, Ny], bool)
    ismissing[int(3*Nx/8): int(Nx*3/4), int(3*Ny/8): int(Ny*3/4)] = True
    f[ismissing] = 0.0

elif (example == 'toy'):
    Nx = 5
    Ny = 5
    f = np.zeros([Nx, Ny])
    f[0,:] = 1.0
    f[1,1:] = 1.0
    f[2,2] = 1.0
    ismissing = np.zeros([Nx, Ny], bool)
    ismissing[1:4,1:4] = True
    f[ismissing] = 0.0

else:
    raise ValueError('Do not know how to intialize when example = ' + example)


# display example to inpaint
f2 = f.copy()
f2[ismissing] = 0.5
plt.figure()
plt.pcolor(f2)
plt.colorbar()
plt.title('initial picture')


"""
Parameters used to build Direct and Adjoint problem
"""

c0 = np.zeros([Nx, Ny])
c0[~ismissing] = 1

alpha = 1e-2

"""
Direct problems: computations
"""

c2_direct_d = np.zeros([Nx, Ny]) + alpha
c2_direct_d[~ismissing] = 0.0
print('Direct Dirichlet: building ...')
tic = time.time()
A_d, b_d = poisson_builder_2D(c0, c2_direct_d, f, 'Dirichlet')
toc = time.time()
print('done in', toc-tic, 'sec')
print('Direct Dirichlet: solving ...')
tic = time.time()
A_d_sp = csc_matrix(A_d)
u_d_1d = spsolve(A_d_sp, b_d)
print('done in', toc-tic, 'sec')
# reshape 1d vector to 2d array
u_d = np.reshape(u_d_1d, [Nx, Ny], 'F')

residues = poisson_verify(c0, c2_direct_d, f, u_d)
print('max(|residues|) = ', np.max(abs(residues)))

c2_direct_n = np.zeros([Nx, Ny]) + alpha
#c2_direct_n[~ismissing] = 0.0
print('Direct Neumann: building ...')
tic = time.time()
A_n, b_n = poisson_builder_2D(c0, c2_direct_n, f, 'Neumann')
toc = time.time()
print('done in', toc-tic, 'sec')
print('Direct Neumann: solving ...')
tic = time.time()
A_n_sp = csc_matrix(A_n)
u_n_1d = spsolve(A_n_sp, b_n)
toc = time.time()
print('done in', toc-tic, 'sec')
# reshape 1d vector to 2d array
u_n = np.reshape(u_n_1d, [Nx, Ny], 'F')

residues = poisson_verify(c0, c2_direct_n, f, u_n)
print('max(|residues|) = ', np.max(abs(residues)))

u_d[~ismissing] = f[~ismissing]
#u_n[~ismissing] = f[~ismissing]
u_dn = u_d - u_n
u_dn[~ismissing] = 0.0

"""
Direct problems: plots
"""

plt.figure()
plt.pcolor(u_d)
plt.colorbar()
plt.title('direct Dirichlet: solution')

plt.figure()
plt.pcolor(u_n)
plt.colorbar()
plt.title('direct Neumann: solution')

plt.figure()
plt.pcolor(u_dn)
plt.colorbar()
plt.title("direct Dirichlet solution 'minus' direct Neumann solution")

"""
Adjoint problems: computations
"""

c2_adjoint_d = np.zeros([Nx, Ny]) + alpha
c2_adjoint_d[~ismissing] = 0.0
print('Adjoint Dirichlet: building ...')
tic = time.time()
A2_d, b2_d = poisson_builder_2D(c0, c2_adjoint_d, u_dn*alpha, 'Dirichlet')
toc = time.time()
print('done in', toc-tic, 'sec')
print('Adjoint Dirichlet: solving ...')
tic = time.time()
A2_d_sp = csc_matrix(A2_d)
v_d_1d = spsolve(A2_d_sp, b2_d)
v_d = np.reshape(v_d_1d, [Nx, Ny], 'F')
toc = time.time()
print('done in', toc-tic, 'sec')

residues = poisson_verify(c0, c2_adjoint_d, u_dn*alpha, v_d)
print('max(|residues|) = ', np.max(abs(residues)))

c2_adjoint_n = np.zeros([Nx, Ny]) + alpha
#c2_adjoint_n[~ismissing] = 0.0
print('Adjoint Neumann: building ...')
tic = time.time()
A2_n, b2_n = poisson_builder_2D(c0, c2_adjoint_n, -u_dn*alpha, 'Neumann')
toc = time.time()
print('done in', toc-tic, 'sec')
print('Adjoint Neumann: solving ...')
tic = time.time()
A2_n_sp = csc_matrix(A2_n)
v_n_1d = spsolve(A2_n_sp, b2_n)
toc = time.time()
print('done in', toc-tic, 'sec')
# reshape 1d vector to 2d array
v_n = np.reshape(v_n_1d, [Nx, Ny], 'F')

residues = poisson_verify(c0, c2_adjoint_n, -u_dn*alpha, v_n)
print('max(|residues|) = ', np.max(abs(residues)))


"""
Adjoint problems: plots
"""

plt.figure()
plt.pcolor(v_d)
plt.colorbar()
plt.title('adjoint Dirichlet: solution')

plt.figure()
plt.pcolor(v_n)
plt.colorbar()
plt.title('adjoint Neumann: solution')

"""
Topological indicator
"""

#v_d[~ismissing] = 0.0
#v_n[~ismissing] = 0.0

topo = np.zeros([Nx, Ny]) + np.nan

for ii in range(1, Nx-1):
    for jj in range(1, Ny-1):

        if not(ismissing[ii, jj]):
            continue

        """
        grad_u_d = 0.5 * np.array([u_d[ii+1, jj] - u_d[ii-1, jj], u_d[ii, jj+1] - u_d[ii, jj-1]])
        grad_v_d = 0.5 * np.array([v_d[ii+1, jj] - v_d[ii-1, jj], v_d[ii, jj+1] - v_d[ii, jj-1]])
        grad_u_n = 0.5 * np.array([u_n[ii+1, jj] - u_n[ii-1, jj], u_n[ii, jj+1] - u_n[ii, jj-1]])
        grad_v_n = 0.5 * np.array([v_n[ii+1, jj] - v_n[ii-1, jj], v_n[ii, jj+1] - v_n[ii, jj-1]])
        """

        """
        grad_u_d = np.array([u_d[ii+1, jj] - u_d[ii, jj], u_d[ii, jj+1] - u_d[ii, jj]])
        grad_v_d = np.array([v_d[ii+1, jj] - v_d[ii, jj], v_d[ii, jj+1] - v_d[ii, jj]])
        grad_u_n = np.array([u_n[ii+1, jj] - u_n[ii, jj], u_n[ii, jj+1] - u_n[ii, jj]])
        grad_v_n = np.array([v_n[ii+1, jj] - v_n[ii, jj], v_n[ii, jj+1] - v_n[ii, jj]])
        """

        grad_u_d = np.array([u_d[ii, jj] - u_d[ii-1, jj], u_d[ii, jj] - u_d[ii, jj-1]])
        grad_v_d = np.array([v_d[ii, jj] - v_d[ii-1, jj], v_d[ii, jj] - v_d[ii, jj-1]])
        grad_u_n = np.array([u_n[ii, jj] - u_n[ii-1, jj], u_n[ii, jj] - u_n[ii, jj-1]])
        grad_v_n = np.array([v_n[ii, jj] - v_n[ii-1, jj], v_n[ii, jj] - v_n[ii, jj-1]])

        """
        grad_u_d = 0.5 * np.array([u_d[ii+1, jj] - u_d[ii, jj] + u_d[ii+1, jj+1] - u_d[ii, jj+1], u_d[ii, jj+1] - u_d[ii, jj] + u_d[ii+1, jj+1] - u_d[ii+1, jj]])
        grad_v_d = 0.5 * np.array([v_d[ii+1, jj] - v_d[ii, jj] + v_d[ii+1, jj+1] - v_d[ii, jj+1], v_d[ii, jj+1] - v_d[ii, jj] + v_d[ii+1, jj+1] - v_d[ii+1, jj]])
        grad_u_n = 0.5 * np.array([u_n[ii+1, jj] - u_n[ii, jj] + u_n[ii+1, jj+1] - u_n[ii, jj+1], u_n[ii, jj+1] - u_n[ii, jj] + u_n[ii+1, jj+1] - u_n[ii+1, jj]])
        grad_v_n = 0.5 * np.array([v_n[ii+1, jj] - v_n[ii, jj] + v_n[ii+1, jj+1] - v_n[ii, jj+1], v_n[ii, jj+1] - v_n[ii, jj] + v_n[ii+1, jj+1] - v_n[ii+1, jj]])
        """

        M = -(grad_u_d.reshape([-1, 1])@grad_v_d.reshape([1, -1]) + grad_u_n.reshape([-1, 1])@grad_v_n.reshape([1, -1]))
        M = 0.5 * (M + M.T)
        eig, _ = la.eigh(M)

        topo[ii, jj] = min(np.real(eig))


#topo[~ismissing] = np.nan

plt.figure()
plt.pcolor(topo)
plt.colorbar()
plt.title('Topological indicator')

plt.figure()
topo_thres = -0.7e-8
is_topo_low = topo < topo_thres
plt.pcolor(is_topo_low + 0.0)
plt.colorbar()
plt.title('Topological indicator (threshold=' + str(topo_thres) + ')')

"""
Inpainting
"""

c2_missing = 1
c2_in = np.zeros([Nx, Ny]) + alpha
c2_in[ismissing] = c2_missing
c2_in[is_topo_low] = alpha
# build
A, b = poisson_builder_2D(c0, c2_in, f, 'Neumann')
# solve
u1d = la.solve(A, b)
# reshape 1d vector to 2d array
u_in = np.reshape(u1d, [Nx, Ny], 'F')

plt.figure()
plt.pcolor(u_in)
plt.colorbar()
plt.title('inpainting: solution')
