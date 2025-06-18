# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 09:20:55 2025

@author: GD5264
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from get_test_image import get_test_image
from poisson_solve_fipy import poisson_solve_fipy
from pcolor_xy import pcolor_xy


"""
The purpose of this script is to implement the inpainting algorithm
described in "A one-shot inpainting algorithm based on the
topological asymptotic analysis" by Auroux and Masmoudi.

This is the same as inpaint4.py buy using 'fipy', a finite volume package.
"""

# close all current plots
plt.close('all')

"""
get test image to inpaint
"""

image_name = 'circle'

f, ismissing = get_test_image(image_name)

Nx, Ny = f.shape

# display test image to inpaint
f2 = f.copy()
f2[ismissing] = 0.5
pcolor_xy(f2, "initial_picture")

"""
Parameters used to build Direct and Adjoint problem
"""

c0 = np.zeros([Nx, Ny])
c0[~ismissing] = 1

alpha = 1e-2

"""
Direct problems: computations
"""

c2_direct_d = np.ones([Nx, Ny])
c2_direct_d[~ismissing] = 0.0
u_d, u_d_x, u_d_y = poisson_solve_fipy(c0, c2_direct_d, f, pin_mask=~ismissing)
# u_d[~ismissing] = f[~ismissing]

c2_direct_n = np.ones([Nx, Ny])
c2_direct_n[~ismissing] = alpha
u_n, u_n_x, u_n_y = poisson_solve_fipy(c0, c2_direct_n, f, pin_mask=None)
# u_n[~ismissing] = f[~ismissing]

u_dn = u_d - u_n
u_dn[~ismissing] = 0.0

"""
Direct problems: plots
"""

pcolor_xy(u_d, "u_d")
pcolor_xy(u_n, "u_n")
pcolor_xy(u_d-u_n, "u_d-u_n")
pcolor_xy(u_dn, "u_dn")

"""
Adjoint problems: computations
"""

c2_adjoint_d = np.ones([Nx, Ny])
c2_adjoint_d[~ismissing] = 0.0
v_d, v_d_x, v_d_y = poisson_solve_fipy(c0, c2_adjoint_d, -u_dn, pin_mask=~ismissing)

c2_adjoint_n = np.ones([Nx, Ny])
c2_adjoint_n[~ismissing] = alpha
v_n, v_n_x, v_n_y = poisson_solve_fipy(c0, c2_adjoint_n, u_dn, pin_mask=None)

"""
Adjoint problems: plots
"""

pcolor_xy(v_d, "v_d")
pcolor_xy(v_n, "v_n")
pcolor_xy(v_d+v_n, "v_d+v_n")

"""
Topological indicator : computations
"""

v_d[~ismissing] = 0.0
v_n[~ismissing] = 0.0


topo_crack = np.zeros([Nx, Ny]) + np.nan
topo_hole = np.zeros([Nx, Ny]) + np.nan

for ii in range(1, Nx-1):
    for jj in range(1, Ny-1):

        if not(ismissing[ii, jj]):
            continue

        """
        # gradient estimated via second-order centered finite-difference
        grad_u_d = 0.5 * np.array([u_d[ii+1, jj] - u_d[ii-1, jj], u_d[ii, jj+1] - u_d[ii, jj-1]])
        grad_v_d = 0.5 * np.array([v_d[ii+1, jj] - v_d[ii-1, jj], v_d[ii, jj+1] - v_d[ii, jj-1]])
        grad_u_n = 0.5 * np.array([u_n[ii+1, jj] - u_n[ii-1, jj], u_n[ii, jj+1] - u_n[ii, jj-1]])
        grad_v_n = 0.5 * np.array([v_n[ii+1, jj] - v_n[ii-1, jj], v_n[ii, jj+1] - v_n[ii, jj-1]])
        """

        """
        # gradient estimated via first‐order forward‐difference
        grad_u_d = np.array([u_d[ii+1, jj] - u_d[ii, jj], u_d[ii, jj+1] - u_d[ii, jj]])
        grad_v_d = np.array([v_d[ii+1, jj] - v_d[ii, jj], v_d[ii, jj+1] - v_d[ii, jj]])
        grad_u_n = np.array([u_n[ii+1, jj] - u_n[ii, jj], u_n[ii, jj+1] - u_n[ii, jj]])
        grad_v_n = np.array([v_n[ii+1, jj] - v_n[ii, jj], v_n[ii, jj+1] - v_n[ii, jj]])
        """

        """
        # gradient estimated via first‐order backward‐difference
        grad_u_d = np.array([u_d[ii, jj] - u_d[ii-1, jj], u_d[ii, jj] - u_d[ii, jj-1]])
        grad_v_d = np.array([v_d[ii, jj] - v_d[ii-1, jj], v_d[ii, jj] - v_d[ii, jj-1]])
        grad_u_n = np.array([u_n[ii, jj] - u_n[ii-1, jj], u_n[ii, jj] - u_n[ii, jj-1]])
        grad_v_n = np.array([v_n[ii, jj] - v_n[ii-1, jj], v_n[ii, jj] - v_n[ii, jj-1]])
        """

        """
        # gradient estimation via averaged forward-difference
        grad_u_d = 0.5 * np.array([u_d[ii+1, jj] - u_d[ii, jj] + u_d[ii+1, jj+1] - u_d[ii, jj+1], u_d[ii, jj+1] - u_d[ii, jj] + u_d[ii+1, jj+1] - u_d[ii+1, jj]])
        grad_v_d = 0.5 * np.array([v_d[ii+1, jj] - v_d[ii, jj] + v_d[ii+1, jj+1] - v_d[ii, jj+1], v_d[ii, jj+1] - v_d[ii, jj] + v_d[ii+1, jj+1] - v_d[ii+1, jj]])
        grad_u_n = 0.5 * np.array([u_n[ii+1, jj] - u_n[ii, jj] + u_n[ii+1, jj+1] - u_n[ii, jj+1], u_n[ii, jj+1] - u_n[ii, jj] + u_n[ii+1, jj+1] - u_n[ii+1, jj]])
        grad_v_n = 0.5 * np.array([v_n[ii+1, jj] - v_n[ii, jj] + v_n[ii+1, jj+1] - v_n[ii, jj+1], v_n[ii, jj+1] - v_n[ii, jj] + v_n[ii+1, jj+1] - v_n[ii+1, jj]])
        """

        # gradient estimation computed by finite volume via 'fipy'
        grad_u_d = np.array([u_d_x[ii, jj], u_d_y[ii, jj]])
        grad_v_d = np.array([v_d_x[ii, jj], v_d_y[ii, jj]])
        grad_u_n = np.array([u_n_x[ii, jj], u_n_y[ii, jj]])
        grad_v_n = np.array([v_n_x[ii, jj], v_n_y[ii, jj]])

        M = -(grad_u_d.reshape([-1, 1])@grad_v_d.reshape([1, -1]) + grad_u_n.reshape([-1, 1])@grad_v_n.reshape([1, -1]))
        M = 0.5 * (M + M.T)
        eig, _ = la.eigh(M)

        topo_crack[ii, jj] = min(np.real(eig))

        topo_hole[ii, jj] = - np.sum(grad_u_d*grad_v_d) + np.sum(grad_u_n*grad_v_n)

# pack
topos = {"crack": topo_crack,
         "hole": topo_hole}

"""
Topological indicator : plots
"""

for topo_type in ["crack", "hole"]:

    topo = topos[topo_type]

    pcolor_xy(topo, f"Topological {topo_type} indicator")

    topo_neg = topo.copy()
    topo_neg[topo_neg>=0] = np.nan
    pcolor_xy(topo_neg, f"Topological {topo_type} indicator (where negative)")

    topo_thres = -2e-4
    is_topo_low = topo < topo_thres
    pcolor_xy(is_topo_low + 0.0, f'Topological {topo_type} indicator (threshold=' + str(topo_thres) + ')')

    """
    Inpainting : computations
    """

    c2_missing = 1
    c2_in = np.zeros([Nx, Ny])
    c2_in[ismissing] = c2_missing
    c2_in[is_topo_low] = alpha

    u_in, u_in_x, u_in_y =  poisson_solve_fipy(c0, c2_in, f, pin_mask=~ismissing)

    """
    Inpainting : plots
    """

    pcolor_xy(u_in, f"Topological {topo_type} inpainting: solution")
