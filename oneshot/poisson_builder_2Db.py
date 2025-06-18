# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:00:33 2022

@author: GD5264
"""

import numpy as np


def poisson_builder_2Db(c0, c2, f, bc):

    """
    Return u, solution of the second order elliptic Partial Differential Equation (PDE)

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

    def avg(a, b):
        # avg_ = 2.0*(a*b)/(a+b)
        avg_ = 0.5*(a+b)
        # avg_ = min(a, b)
        return avg_

    exists = np.abs(c0) > 1e-16

    A = np.zeros((N, N))

    for ii in range(Nx):
        for jj in range(Ny):
            A[mm(ii, jj), mm(ii, jj)] = c0[ii, jj]
            b[mm(ii, jj)] = f[ii, jj]

    for ii in range(Nx-1):
        for jj in range(Ny-1):

            if (bc == 'Dirichlet') and exists[ii, jj]:
                pass
            else:
                A[mm(ii, jj), mm(ii, jj)] += c2[ii, jj]
                A[mm(ii, jj), mm(ii, jj+1)] -= 0.5*c2[ii, jj]
                A[mm(ii, jj), mm(ii+1, jj)] -= 0.5*c2[ii, jj]

            if (bc == 'Dirichlet') and exists[ii, jj+1]:
                pass
            else:
                A[mm(ii, jj+1), mm(ii, jj+1)] += c2[ii, jj+1]
                A[mm(ii, jj+1), mm(ii, jj)] -= 0.5*c2[ii, jj+1]
                A[mm(ii, jj+1), mm(ii+1, jj+1)] -= 0.5*c2[ii, jj+1]

            if (bc == 'Dirichlet') and exists[ii+1, jj]:
                pass
            else:
                A[mm(ii+1, jj), mm(ii+1, jj)] += c2[ii+1, jj]
                A[mm(ii+1, jj), mm(ii, jj)] -= 0.5*c2[ii+1, jj]
                A[mm(ii+1, jj), mm(ii+1, jj+1)] -= 0.5*c2[ii+1, jj]

            if (bc == 'Dirichlet') and exists[ii+1, jj+1]:
                pass
            else:
                A[mm(ii+1, jj+1), mm(ii+1, jj+1)] += c2[ii+1, jj+1]
                A[mm(ii+1, jj+1), mm(ii, jj+1)] -= 0.5*c2[ii+1, jj+1]
                A[mm(ii+1, jj+1), mm(ii+1, jj)] -= 0.5*c2[ii+1, jj+1]

    return A, b


if (__name__ == '__main__'):

    import scipy.linalg as la

    from poisson_verify import poisson_verify

    Nx = 9
    Ny = 9

    c0 = np.zeros([Nx, Ny])
    c0[1, 1] = 1.0
    c2 = np.ones([Nx, Ny])
    f = np.ones([Nx, Ny])
    bc = 'Neumann'
    A, b = poisson_builder_2Db(c0, c2, f, bc)

    u_1d = la.solve(A, b)

    u = np.reshape(u_1d, [Nx, Ny], 'F')

    res = poisson_verify(c0, c2, f, u)
