# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:00:33 2022

@author: GD5264
"""

import numpy as np


def poisson_builder_2D(c0, c2, f, bc):

    """
    Return u, solution of the second order elliptic Partial Differential Equation (PDE)

        - div[c2(x,y) * grad[u(x,y)]] + c0(x,y) * u(x,y) = f(x,y)

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
        avg_ = 0.5*(a+b)
        """
        When the operator d/dx[c * d/dx[u(x)]] is discretized by finite volume
        as described in https://www.math.univ-toulouse.fr/~fboyer/_media/exposes/mexique2013.pdf
        then avg(a,b) should be the harmonic mean.
        Remark: Apparently the harmonic mean is valid in 1D (however, need to double check that harmonic mean is also valid in 2D)
        """
        # avg_ = 2.0*(a*b)/(a+b)
        return avg_

    exists = np.abs(c0) > 1e-16

    A = np.zeros((N, N))

    for jj in range(Ny):
        for ii in range(Nx):
            A[mm(ii, jj), mm(ii, jj)] = c0[ii, jj]
            b[mm(ii, jj)] = f[ii, jj]

    for jj in range(Ny):
        for ii in range(Nx):

            if (bc == 'Dirichlet'):
                if (exists[ii, jj]):
                    continue

            # check left cell
            if ((ii-1) >= 0):
                k = avg(c2[ii-1, jj], c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii-1, jj)] -= k

            # check right cell
            if ((ii+1) <= (Nx-1)):
                k = avg(c2[ii+1, jj], c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii+1, jj)] -= k

            # check lower cell
            if ((jj-1) >= 0):
                k = avg(c2[ii, jj-1], c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii, jj-1)] -= k

            # check upper cell
            if ((jj+1) <= (Ny-1)):
                k = avg(c2[ii, jj+1], c2[ii, jj])
                A[mm(ii, jj), mm(ii, jj)] += k
                A[mm(ii, jj), mm(ii, jj+1)] -= k

    return A, b


if (__name__ == '__main__'):

    import scipy.linalg as la

    from poisson_verify import poisson_verify

    Nx = 9
    Ny = 9

    c0 = np.zeros([Nx, Ny])
    c0[Nx//2, Ny//2] = 1.0
    c2 = np.ones([Nx, Ny])
    f = np.ones([Nx, Ny])
    bc = 'Neumann'
    A, b = poisson_builder_2D(c0, c2, f, bc)

    u = la.solve(A, b)

    u = np.reshape(u, [Nx, Ny], 'F')

    res = poisson_verify(c0, c2, f, u)
