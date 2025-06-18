# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:00:33 2022

@author: GD5264
"""

import numpy as np
import scipy.sparse as sp


def poisson_builder_1D(c0, c2, f, bc):

    """
    Return A and b, discretization of the second order Two Point Boundary Value Problem (TPBVP)

        - d/dx[c2(x) * d/dx[u(x)]] + c0(x) * u(x) = f(x)

    on a rectangular domain with Neumann boundary conditions.

    When c2*c0 > 0, the TPBVP is called 'Screened Poisson equation'
    When c2*c0 < 0, the TPBVP is called 'Helmholtz equation'

    This TPBVP is discretized by (cell centered) finite difference.
    """

    # sanity check
    if not(c0.shape == f.shape):
        raise ValueError("c0 and f must have same shape")
    if not(c2.shape == f.shape):
        raise ValueError("c2 and f must have same shape")
    if not(bc in ['Dirichlet', 'Neumann']):
        raise ValueError('cannot have bc = ' + bc)

    N = c0.shape[0]

    # initialization
    b = np.zeros(N) + np.nan

    def avg(a, b):
        avg_ = 0.5*(a+b)
        """
        When the operator d/dx[c * d/dx[u(x)]] is discretized by finite volume
        as described in https://www.math.univ-toulouse.fr/~fboyer/_media/exposes/mexique2013.pdf
        then avg(a,b) should be the harmonic mean
        """
        avg_ = 2.0*(a*b)/(a+b)
        return avg_

    exists = np.abs(c0) > 1e-16

    # A = sp.coo_matrix((N, N))
    A = np.zeros((N, N))

    for ii in range(N):

        A[ii, ii] = c0[ii]
        b[ii] = f[ii]

        if (bc == 'Dirichlet'):
            if (exists[ii]):
                continue

        # check left cell
        if ((ii-1) >= 0):
            k = avg(c2[ii-1], c2[ii])
            A[ii, ii] += k
            A[ii, ii-1] -= k

        # check right cell
        if ((ii+1) <= (N-1)):
            k = avg(c2[ii+1], c2[ii])
            A[ii, ii] += k
            A[ii, ii+1] -= k

    return A, b
