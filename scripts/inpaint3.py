# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:15:17 2021

@authors: Axel Breuer and Didier Auroux
"""


import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

#from poisson_solve import poisson_solve
#from poisson_solve2 import poisson_solve2
from poisson_solve2 import poisson_solve2 as poisson_solve3

"""
The purpose of this script is to implement the inpainting algorithm
described in "A one-shot inpainting algorithm based on the
topological asymptotic analysis" by Auroux and Masmoudi.
"""

plt.close('all')


"""
build example to inpaint
"""

Nx = 128
Ny = 128
example = 'circle'

if (example == 'circle'):

    f = np.zeros([Nx, Ny])
    for ii in range(Nx+1):
        for jj in range(Ny+1):
            if (((ii-Nx//2)**2+(jj-Ny//2)**2) < (min(Nx,Ny)**2)/16):
                f[ii, jj] = 1.0
    ismissing = np.zeros([Nx, Ny], bool)
    ismissing[Nx//2: int(Nx*7/8)+1, Ny//2: int(Ny*7/8)+1] = True
    f[ismissing] = 0.0

elif (example == 'square'):

    f = np.zeros([Nx, Ny])
    f[:Nx//2, :Ny//2] = 1.0
    ismissing = np.zeros([Nx, Ny], bool)
    ismissing[int(3*Nx/8): int(Nx*3/4), int(3*Ny/8): int(Ny*3/4)] = True
    f[ismissing] = 0.0

else:

    raise ValueError('Do not know how to intialize when example = ' + example)

"""
# display example to inpaint
f2 = f.copy()
f2[ismissing] = 0.5
plt.figure()
plt.pcolor(f2)
plt.colorbar()
plt.title('initial picture')
"""

"""
Parameters used to build Direct and Adjoint problem
"""

alpha = 1e-4

c0 = np.ones([Nx, Ny])
c0[ismissing] = 0.0

c2_direct_d = np.zeros([Nx, Ny]) + 1.0
#c2_direct_d[ismissing] = 1.0

c2_direct_n = np.zeros([Nx, Ny]) + alpha

c2_adjoint_d = np.zeros([Nx, Ny]) + 1.0
#c2_adjoint_d[ismissing] = 1.0

c2_adjoint_n = np.zeros([Nx, Ny]) + alpha


"""
Direct problems
"""

print('solve Direct Dirichlet ...')
tic = time.time()
u_d, _, _ = poisson_solve3(c0, c2_direct_d, f)
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.pcolor(u_d)
plt.colorbar()
plt.title('direct Dirichlet: solution')

print('solve Direct Neumann ...')
tic = time.time()
u_n, _, _ = poisson_solve3(c0, c2_direct_n, f)
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.pcolor(u_n)
plt.colorbar()
plt.title('direct Neumann: solution')

u_d[~ismissing] = f[~ismissing]
u_n[~ismissing] = f[~ismissing]
u_dn = u_d - u_n
u_dn[~ismissing] = 0.0

plt.figure()
plt.pcolor(u_dn)
plt.colorbar()
plt.title("direct Dirichlet solution 'minus' direct Neumann solution")

"""
Adjoint problems
"""

print('solve Adjoint Dirichlet ...')
tic = time.time()
v_d, _, _ = poisson_solve3(c0, c2_adjoint_d, -u_dn)
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.pcolor(v_d)
plt.colorbar()
plt.title('adjoint Dirichlet: solution')

print('solve Adjoint Neumann ...')
tic = time.time()
v_n, _, _ = poisson_solve3(c0, c2_adjoint_n, 1e-4*u_dn)
toc = time.time()
print('done in', toc-tic, 'sec')

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

        grad_u_d = 0.5 * np.array([u_d[ii+1, jj] - u_d[ii-1, jj],
                                   u_d[ii, jj+1] - u_d[ii, jj-1]])
        grad_v_d = 0.5 * np.array([v_d[ii+1, jj] - v_d[ii-1, jj],
                                   v_d[ii, jj+1] - v_d[ii, jj-1]])
        grad_u_n = 0.5 * np.array([u_n[ii+1, jj] - u_n[ii-1, jj],
                                   u_n[ii, jj+1] - u_n[ii, jj-1]])
        grad_v_n = 0.5 * np.array([v_n[ii+1, jj] - v_n[ii-1, jj],
                                   v_n[ii, jj+1] - v_n[ii, jj-1]])

        M = -(grad_u_d.reshape([-1, 1])@grad_v_d.reshape([1, -1]) + grad_u_n.reshape([-1, 1])@grad_v_n.reshape([1, -1]))
        M = 0.5 * (M + M.T)
        eig, _ = la.eigh(M)

        topo[ii, jj] = min(np.real(eig))

plt.figure()
plt.pcolor(topo)
plt.colorbar()
plt.title('Topological indicator')

plt.figure()
topo_thres = - 0.8e-4
is_topo_low = topo < topo_thres
plt.pcolor(is_topo_low + 0.0)
plt.colorbar()
plt.title('Topological indicator (threshold=' + str(topo_thres) + ')')

"""
Inpainting
"""

c2_in = c2_direct_n.copy()
c2_in[is_topo_low] = alpha
u_in, _, _ = poisson_solve3(c0, c2_in, f)

plt.figure()
plt.pcolor(u_in)
plt.colorbar()
plt.title('inpainting: solution')
