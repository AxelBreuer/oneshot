# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:00:33 2022

@author: GD5264
"""

import numpy as np
import scipy.sparse as sp


def poisson_builder_2D(c0, c2, f, bc):

    """
    Return A and b, discretization of the second order elliptic PDE

        - div[c2(x,y) * grad[u(x,y)]] + c0(x,y) u(x,y) = f(x,y)

    on a rectangular domain with Neumann boundary conditions.

    When c2*c0>0, the PDE is called 'Screened Poisson equation'
    When c2*c0<0, the PDE is called 'Helmholtz equation'

    This PDE is discretized by finite volume.

    The operator div[c grad[u]] is discretized by finite volume
    as described in https://www.math.univ-toulouse.fr/~fboyer/_media/exposes/mexique2013.pdf
    """

    # sanity check
    if not(c0.shape == f.shape):
        raise ValueError("c0 and f must have same shape")
    if not(c2.shape == f.shape):
        raise ValueError("c2 and f must have same shape")
    if not(bc in ['Dirichlet', 'Neumann']):
        raise ValueError('cannot have bc = ' + bc)

    Nx, Ny = c0.shape

    # initialization
    N = Nx * Ny
    b = np.zeros(N) + np.nan

    def mm(ii, jj):
        return jj*Nx+ii

    exists = np.abs(c0) > 1e-16

    #A = sp.coo_matrix((N, N))
    A = np.zeros((N, N))

    for jj in range(Ny):
        for ii in range(Nx):
            A[mm(ii, jj), mm(ii, jj)] = c0[ii, jj]
            b[mm(ii, jj)] = f[ii, jj]

            if (bc == 'Dirichlet'):
                if (exists[ii, jj]):
                    continue

            # check left cell
            if ((ii-1) < 0):
                pass
            elif (exists[ii-1, jj]):
                if (bc == 'Dirichlet'):
                    k = c2[ii, jj]
                elif (bc == 'Neumann'):
                    k = 2.0 * (c2[ii-1, jj]*c2[ii, jj])/(c2[ii-1, jj]+c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii-1, jj)] -= k
            else:
                k = 2.0 * (c2[ii-1, jj]*c2[ii, jj])/(c2[ii-1, jj]+c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii-1, jj)] -= k

            # check right cell
            if ((ii+1) > (Nx-1)):
                pass
            elif (exists[ii+1, jj]):
                if (bc == 'Dirichlet'):
                    k = c2[ii, jj]
                elif (bc == 'Neumann'):
                    k = 2.0 * (c2[ii+1,jj]*c2[ii, jj])/(c2[ii+1, jj]+c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii+1, jj)] -= k
            else:
                k = 2.0 * (c2[ii+1,jj]*c2[ii, jj])/(c2[ii+1, jj]+c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii+1, jj)] -= k

            # check lower cell
            if ((jj-1) < 0):
                pass
            elif (exists[ii, jj-1]):
                if (bc == 'Dirichlet'):
                    k = c2[ii, jj]
                elif (bc == 'Neumann'):
                    k = 2.0 * (c2[ii, jj-1]*c2[ii, jj])/(c2[ii, jj-1]+c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii, jj-1)] -= k
            else:
                k = 2.0 * (c2[ii, jj-1]*c2[ii, jj])/(c2[ii, jj-1]+c2[ii, jj])
                #k = 0.5*(c2[ii, jj-1]+c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii, jj-1)] -= k

            # check upper cell
            if ((jj+1) > (Ny-1)):
                pass
            elif (exists[ii, jj+1]):
                if (bc == 'Dirichlet'):
                    k = c2[ii, jj]
                elif (bc == 'Dirichlet'):
                    k = 2.0 * (c2[ii, jj+1]*c2[ii, jj])/(c2[ii, jj+1]+c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii, jj+1)] -= k
            else:
                k = 2.0 * (c2[ii, jj+1]*c2[ii, jj])/(c2[ii, jj+1]+c2[ii, jj])
                #k = 0.5*(c2[ii, jj+1]+c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii, jj+1)] -= k

    return A, b
