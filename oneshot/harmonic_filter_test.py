# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:14:50 2022

@author: GD5264
"""

from harmonic_filter import harmonic_filter

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import time

from poisson_builder_1D import poisson_builder_1D

plt.close('all')

ys = np.array(list(np.ones(100)) + list(1 + np.cumsum(np.ones(200)/100)))
ys[50:250] = np.nan

N = len(ys)

plt.figure()
cs = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
for c in cs:
    l2s = c*np.ones(N-1)
    harmonic_ys, obj = harmonic_filter(ys, l2s)
    plt.plot(harmonic_ys)
harmonic0_ys, obj = harmonic_filter(ys, l2s, interp_p=True)
plt.plot(harmonic0_ys, 'k')
plt.plot(ys, 'ro')
plt.legend(cs + ['tv0_y', 'y'])


c = 0.1
l2s = c*np.ones(N-1)
objs = np.zeros(N-1)
harmonic0_ys, obj = harmonic_filter(ys, l2s, interp_p=True)

plt.figure()
plt.plot(ys, 'o')
plt.plot(harmonic0_ys)

for ii in range(N-1):

    l2s_tmp = l2s.copy()
    l2s_tmp[ii] = 0
    print(ii, '->', l2s_tmp)

    harmonic_ys, obj = harmonic_filter(ys, l2s_tmp)

    objs[ii] = obj

    if (ii%20 == 0) and (ii>=50):
        plt.plot(harmonic_ys)
        plt.legend(['input', 'harmonic0', 'harmonic'])

plt.figure()
plt.plot(ys, 'o')
plt.plot(harmonic0_ys)
plt.plot(harmonic_ys)
plt.legend(['input', 'harmonic0', 'harmonic'])

plt.figure()
plt.plot(objs)

"""
Parameters used to build Direct and Adjoint problem
"""

ismissing = np.zeros(N, bool)
ismissing[50:250] = True

f = ys.copy()
f[ismissing] = 0.0


c0 = np.zeros(N)
c0[~ismissing] = 1.0

c2_direct_d = np.zeros(N)
c2_direct_d[ismissing] = 1.0

c2_direct_n = np.zeros(N) + 1e-4
c2_direct_n[ismissing] = 1.0


c2_adjoint_d = np.zeros(N)
c2_adjoint_d[ismissing] = 1.0

c2_adjoint_n = np.zeros(N) + 1e-4
c2_adjoint_n[ismissing] = 1.0

"""
Direct problems
"""

print('solve Direct Dirichlet ...')

tic = time.time()
# build
A_d, b_d = poisson_builder_1D(c0, c2_direct_d, f, 'Dirichlet')
# solve
u_d = la.solve(A_d, b_d)
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.plot(u_d)
plt.title('direct Dirichlet: solution')

print('solve Direct Neumann ...')
tic = time.time()
# build
A_n, b_n = poisson_builder_1D(c0, c2_direct_n, f, 'Neumann')
# solve
u_n = la.solve(A_n, b_n)
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.plot(u_n)
plt.title('direct Neumann: solution')

"""
Adjoint problems
"""

u_dn = u_d - u_n
u_dn[~ismissing] = 0


print('solve Adjoint Dirichlet ...')
tic = time.time()
# build
AA_d, bb_d = poisson_builder_1D(c0, c2_adjoint_d, -u_dn, 'Dirichlet')
# solve
v_d = la.solve(AA_d, bb_d)
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.plot(v_d)
plt.title('adjoint Dirichlet: solution')

print('solve Adjoint Neumann ...')
tic = time.time()
# build
AA_n, bb_n = poisson_builder_1D(c0, c2_adjoint_n, u_dn, 'Neumann')
# solve
v_n = la.solve(AA_n, bb_n)
toc = time.time()
print('done in', toc-tic, 'sec')

plt.figure()
plt.plot(v_n)
plt.title('adjoint Neumann: solution')

topo = np.zeros(N) + np.nan

for ii in range(1, N):

    if not(ismissing[ii]):
        continue

    grad_u_d = 0.5 * (u_d[ii+1] - u_d[ii-1])
    grad_v_d = 0.5 * (v_d[ii+1] - v_d[ii-1])
    grad_u_n = 0.5 * (u_n[ii+1] - u_n[ii-1])
    grad_v_n = 0.5 * (v_n[ii+1] - v_n[ii-1])

    M = -(grad_u_d*grad_v_d + grad_u_n*grad_v_n)

    topo[ii] = M

plt.figure()
plt.plot(topo, 'o-')
plt.title('topological indicator')
