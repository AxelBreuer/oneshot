# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:15:17 2021

@author: Axel Breuer (with guidance by Didier Auroux)
"""


import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from poisson_solve import poisson_solve
from poisson_verify import poisson_verify


"""
The purpose of this script is to implement the inpainting algorithm
described in "A one-shot inpainting algorithm based on the
topological asymptotic analysis" by Auroux and Masmoudi.
"""

def grad(v, tau=1.0):#np.sqrt(2)-1):
    gradx = (v[2,:] - v[0,:]) @ np.array([(1-tau)/2, tau, (1-tau)/2]) * 0.5
    grady = (v[:,2] - v[:,0]) @ np.array([(1-tau)/2, tau, (1-tau)/2]) * 0.5
    return np.array([gradx, grady])


plt.close('all')


"""
build example to inpaint
"""

Nx = 50
Ny = 50

# circle
"""
f = np.zeros([Nx, Ny])
for ii in range(Nx+1):
    for jj in range(Ny+1):
        if (((ii-Nx//2)**2+(jj-Ny//2)**2) < (min(Nx,Ny)**2)/16):
            f[ii, jj] = 1.0
ismissing = np.zeros([Nx, Ny], bool)
ismissing[Nx//2: int(Nx*7/8)+1, Ny//2: int(Ny*7/8)+1] = True
f[ismissing] = 0.0
"""


# square
f = np.zeros([40, 40])
f[:20, :20] = 1.0
ismissing = np.zeros([40, 40], bool)
ismissing[10:30, 10:30] = True
f[ismissing] = 0.0
Nx, Ny = f.shape


f2 = f.copy()
f2[ismissing] = 0.0
plt.figure()
plt.pcolor(f2)
plt.colorbar()
plt.title('initial picture')

"""
Parameters used to build Direct and Adjoint problem
"""

alpha = 1e-4

"""
Direct problems
"""
print('solve Direct Dirichlet ...')
c0_d = np.zeros([Nx, Ny])
c0_d[~ismissing] = 1.0
c2_d = np.zeros([Nx, Ny])
c2_d[ismissing] = alpha # 1.0
tic = time.time()
u_d, _, _ = poisson_solve(c0_d, c2_d, f)
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.pcolor(u_d)
plt.colorbar()
plt.title('direct Dirichlet: solution')

res_d = poisson_verify(c0_d, c2_d, f, u_d)

plt.figure()
plt.pcolor(res_d)
plt.colorbar()
plt.title('direct Dirichlet: residues')

print('solve Direct Neumann ...')
c0_n = np.zeros([Nx, Ny])
c0_n[~ismissing] = 1.0
c2_n = np.zeros([Nx, Ny]) + alpha
c2_n[ismissing] = alpha # 1
tic = time.time()
u_n, _, _ = poisson_solve(c0_n, c2_n, f)
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.pcolor(u_n)
plt.colorbar()
plt.title('direct Neumann: solution')

res_n = poisson_verify(c0_n, c2_n, f, u_n)

plt.figure()
plt.pcolor(res_n)
plt.colorbar()
plt.title('direct Neumann: residues')

u_d[~ismissing] = f[~ismissing]
u_n[~ismissing] = f[~ismissing]

plt.figure()
plt.pcolor(u_d-u_n)
plt.colorbar()
plt.title("direct Dirichlet solution 'minus' direct Neumann solution")

"""
Adjoint problems
"""

print('solve Adjoint Dirichlet ...')
c0_d = np.zeros([Nx, Ny])
c0_d[~ismissing] = 1.0
c2_d = np.zeros([Nx, Ny])
c2_d[ismissing] = 1.0
tic = time.time()
v_d, _, _ = poisson_solve(c0_d, c2_d, -(u_d-u_n))
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.pcolor(v_d)
plt.colorbar()
plt.title('adjoint Dirichlet: solution')

res_d = poisson_verify(c0_d, c2_d, -(u_d-u_n), v_d)
del c0_d, c2_d

plt.figure()
plt.pcolor(res_d)
plt.colorbar()
plt.title('adjoint Dirichlet: residues')

print('solve Adjoint Neumann ...')
c0_n = np.zeros([Nx, Ny])
c0_n[~ismissing] = 1.0
c2_n = np.zeros([Nx, Ny]) + alpha
c2_n[ismissing] = 1.0
tic = time.time()
v_n, _, _ = poisson_solve(c0_n, c2_n, u_d - u_n)
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.pcolor(v_n)
plt.colorbar()
plt.title('adjoint Neumann: solution')

res_n = poisson_verify(c0_n, c2_n, u_d - u_n, v_n)
del c0_n, c2_n

plt.figure()
plt.pcolor(res_n)
plt.colorbar()
plt.title('adjoint Neumann: residues')

"""
Topological indicator
"""

#v_d[~ismissing] = 0.0
#v_n[~ismissing] = 0.0

topo_min = np.zeros([Nx, Ny]) + np.nan
topo_max = np.zeros([Nx, Ny]) + np.nan

for ii in range(1, Nx-1):
    for jj in range(1, Ny-1):
        if not(ismissing[ii, jj]):
            continue

        """
        grad_u_d = 0.5*np.array([u_d[ii+1, jj] - u_d[ii-1, jj], u_d[ii, jj+1] - u_d[ii, jj-1]])
        grad_v_d = 0.5*np.array([v_d[ii+1, jj] - v_d[ii-1, jj], v_d[ii, jj+1] - v_d[ii, jj-1]])
        grad_u_n = 0.5*np.array([u_n[ii+1, jj] - u_n[ii-1, jj], u_n[ii, jj+1] - u_n[ii, jj-1]])
        grad_v_n = 0.5*np.array([v_n[ii+1, jj] - v_n[ii-1, jj], v_n[ii, jj+1] - v_n[ii, jj-1]])
        """

        grad_u_d = grad(u_d[ii-1:ii+2, jj-1:jj+2])
        grad_v_d = grad(v_d[ii-1:ii+2, jj-1:jj+2])
        grad_u_n = grad(u_n[ii-1:ii+2, jj-1:jj+2])
        grad_v_n = grad(v_n[ii-1:ii+2, jj-1:jj+2])

        M = (grad_u_d.reshape([-1, 1])@grad_v_d.reshape([1, -1]) + grad_u_n.reshape([-1, 1])@grad_v_n.reshape([1, -1]))

        M = - 0.5*(M + M.T)
        eig, _ = la.eigh(M)

        topo_min[ii, jj] = np.min(eig)
        topo_max[ii, jj] = np.max(eig)

plt.figure()
plt.pcolor(topo_min)
plt.colorbar()
plt.title('Topological indicator')

plt.figure()
thres = -2e-11
is_topo_low = topo_min < thres
plt.pcolor(is_topo_low + 0.0)
plt.colorbar()
plt.title('Topological indicator (threshold=' + str(thres) + ')')

"""
Inpainting
"""

c0_in = np.zeros([Nx, Ny])
c0_in[~ismissing] = 1.0
c0_in[is_topo_low] = 0.0
c2_in = np.zeros([Nx, Ny]) + alpha
c2_in[ismissing] = 1.0
c2_in[is_topo_low] = alpha
u_in, _, _ = poisson_solve(c0_in, c2_in, f)

plt.figure()
plt.pcolor(u_in)
plt.colorbar()
plt.title('inpainting: solution')

res_in = poisson_verify(c0_in, c2_in, f, u_in)

plt.figure()
plt.pcolor(res_n)
plt.colorbar()
plt.title('inpainting: residues')
