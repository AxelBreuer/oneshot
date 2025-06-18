# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:38:25 2021

@authors: Axel Breuer and Didier Auroux
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse.linalg as sla


def poisson_solve4(c0, c2, f, solver_type='sparse'):

    """
    Return u, solution of the second order elliptic PDE

        - div[c2(x,y) * grad[u(x,y)]] + c0(x,y) u(x,y) = f(x,y)

    on a rectangular domain with Neumann boundary conditions.

    When c2*c0>0, the PDE is called 'Screened Poisson equation'
    When c2*c0<0, the PDE is called 'Helmholtz equation'

    This PDE is discretized by finite difference.

    The operator div[c grad[u]] is discretized by finite difference
    as described in "Elliptic Differential Equations" by Hackbusch (section 5.1.4)
    using harmonic mean, instead of arithmetic mean, as advocated in
    "Finite Difference Computing with PDEs" by Langtangen (section 2.7.3)

    The Neumann boundary condition is discretized by symmetric difference
    as described in "Elliptic Differential Equations" by Hackbusch (section 4.7.2).

    The constructed 'sparse' linear system is then solved by a 'sparse' or 'dense' solver (default is 'sparse')

    To verify the accuracy of the calculated solution u, you can use poisson_verify()
    """

    # sanity check
    if not(c0.shape == f.shape):
        raise ValueError("c0 and f must have same shape")
    if not(c2.shape == f.shape):
        raise ValueError("c2 and f must have same shape")
    if not(solver_type in ['sparse', 'dense', None]):
        raise ValueError("solver_type must be either 'sparse' or 'dense'")

    """
    The construction of the sparse linear system is described in
    "Finite Difference Computing with PDEs" by Langtangen and Linge
    (section 3.6.7)
    """

    """
    https://www.math.univ-toulouse.fr/~fboyer/_media/exposes/mexique2013.pdf
    """

    Nx, Ny = c0.shape

    # initialization
    N = Nx * Ny
    b = np.zeros(N) + np.nan

    def mm(ii, jj):
        return jj*Nx+ii

    ismissing = np.abs(c0)<1e-16

    A = sp.coo_matrix((Nx, Nx))

    for jj in range(Ny):
        for ii in range(Nx):
            A[mm(ii, jj), mm(ii,jj)] = c0[ii, jj]
            b[mm(ii, jj)] = f[ii, jj]
            if ~ismissing[ii, jj]:
                continue

            if




            if (jj == 0):
                if (ii == 0):
                    A[mm(0, 0), mm(0, 0)] += 2*c2[0, 0]
                    A[mm(0, 0), mm(0, 1)] = -c2[0, 1]
                    A[mm(0, 0), mm(1, 0)] = -c2[1, 0]
                elif (ii < (Nx-1)):
                    A[mm(ii, 0), mm(ii-1, 0)] = -c2[ii-1, 0]
                    A[mm(ii, 0), mm(ii, 0)] += 3*c2[ii, 0]
                    A[mm(ii, 0), mm(ii, 1)] = -c2[ii, 1]
                    A[mm(ii, 0), mm(ii+1, 0)] = -c2[ii+1, 0]
                else:
                    A[mm(Nx-1, 0), mm(Nx-2, 1)] = -c2[Nx-2, 1]
                    A[mm(Nx-1, 0), mm(Nx-1, 0)] += 2*c2[Nx-1, 0]
                    A[mm(Nx-1, 0), mm(Nx-1, 1)] = -c2[Nx-1, 1]

            elif (jj < (Ny-1)):
                if (ii == 0):
                    A[mm(0, jj), mm(0, jj-1)] = -c2[0, jj-1]
                    A[mm(0, jj), mm(0, jj)] += 3*c2[0, jj]
                    A[mm(0, jj), mm(0, jj+1)] = -c2[0, jj+1]
                    A[mm(0, jj), mm(1, jj)] = -c2[1, jj]
                elif (ii < (Nx-1)):
                    A[mm(ii, jj), mm(ii-1, jj)] = -c2[ii-1, jj]
                    A[mm(ii, jj), mm(ii, jj-1)] = -c2[ii, jj-1]
                    A[mm(ii, jj), mm(ii, jj)] += 4*c2[ii, jj]
                    A[mm(ii, jj), mm(ii, jj+1)] = -c2[ii, jj+1]
                    A[mm(ii, jj), mm(ii+1, jj)] = -c2[ii+1, jj]
                else:
                    A[mm(Nx-1, jj), mm(Nx-2, jj)] = -c2[Nx-2, jj]
                    A[mm(Nx-1, jj), mm(Nx-1, jj-1)] = -c2[Nx-1, jj-1]
                    A[mm(Nx-1, jj), mm(Nx-1, jj)] += 3*c2[Nx-1, jj]
                    A[mm(Nx-1, jj), mm(Nx-1, jj+1)] = -c2[Nx-1, jj+1]

            else:
                if (ii == 0):
                    A[mm(0, Ny-1), mm(0, Ny-2)] = -c2[0, Ny-2]
                    A[mm(0, Ny-1), mm(0, Ny-1)] = 2*c2[0, Ny-1]
                    A[mm(0, Ny-1), mm(1, Ny-1)] = -c2[1, Ny-1]
                elif (ii < (Nx-1)):
                    A[mm(ii, Ny-1), mm(ii, Ny-2)] = 3*c2[ii, Ny-2]
                    A[mm(ii, Ny-1), mm(ii, Ny-1)] = -c2[ii, Ny-1]
                    A[mm(ii, Ny-1), mm(ii-1, Ny-1)] = -c2[ii-1, Ny-1]
                    A[mm(ii, Ny-1), mm(ii+1, Ny-1)] = -c2[ii+1, Ny-1]
                else:
                    A[mm(Nx-1, Ny-1), mm(Nx-2, Ny-1)] = -c2[Nx-2, Ny-1]
                    A[mm(Nx-1, Ny-1), mm(Nx-1, Ny-2)] = -c2[Nx-1, Ny-2]
                    A[mm(Nx-1, Ny-1), mm(Nx-1, Ny-1)] = 2*c2[Nx-1, Ny-1]








    lower_offset = 1
    lower2_offset = Nx

    """
    jj = 0 (lower boundary)
    """

    # ii = 0
    upper2[mm(0, 0)] = -2.0 * c2[0, 0]
    upper[mm(0, 0)] = -2.0 * c2[0, 0]
    main[mm(0, 0)] = 4.0 * c2[0, 0] + c0[0, 0]

    # 1 <= ii <= Nx-2
    upper2[mm(1, 0): mm(Nx-1, 0)] = -2.0 * c2[1:Nx-1, 0]
    upper[mm(1, 0): mm(Nx-1, 0)] = -c2[1:Nx-1, 0]
    main[mm(1, 0):mm(Nx-1, 0)] = 4.0 * c2[1:Nx-1, 0] + c0[1:Nx-1, 0]
    lower[mm(1, 0)-lower_offset: mm(Nx-1, 0)-lower_offset] = -c2[1:Nx-1, 0]

    # ii = Nx-1
    upper2[mm(Nx-1, 0)] = -2.0 * c2[Nx-1, 0]
    main[mm(Nx-1, 0)] = 4.0 * c2[Nx-1, 0] + c0[Nx-1, 0]
    lower[mm(Nx-1, 0)-lower_offset] = -2.0 * c2[Nx-1, 0]

    # rhs
    b[mm(0, 0): mm(Nx, 0)] = f[:, 0]

    """
    1 <= jj <= Ny-2
    """

    for jj in range(1, Ny-1):

        # ii = 0 (left boundary)
        upper2[mm(0, jj)] = -c2[0, jj]
        upper[mm(0, jj)] = -2.0 * c2[0, jj]
        main[mm(0, jj)] = 4.0 * c2[0, jj] + c0[0, jj]
        lower2[mm(0, jj)-lower2_offset] = -c2[0, jj]

        # 1 <= ii <= Nx-2 (interior points)
        #upper2[mm(1, jj): mm(Nx-1, jj)] = -0.5*(c2[1:Nx-1, jj] + c2[1:Nx-1, jj+1])
        upper2[mm(1, jj): mm(Nx-1, jj)] = -hmean(c2[1:Nx-1, jj], c2[1:Nx-1, jj+1])
        #upper[mm(1, jj): mm(Nx-1, jj)] = -0.5*(c2[1:Nx-1, jj] + c2[2:Nx, jj])
        upper[mm(1, jj): mm(Nx-1, jj)] = -hmean(c2[1:Nx-1, jj], c2[2:Nx, jj])
        #main[mm(1, jj):mm(Nx-1, jj)] = (2*c2[1:Nx-1, jj] + 0.5*c2[0:Nx-2, jj] + 0.5*c2[2:Nx, jj] + 0.5*c2[1:Nx-1, jj-1] + 0.5*c2[1:Nx-1, jj+1]) + c0[1:Nx-1, jj]
        main[mm(1, jj): mm(Nx-1, jj)] = hmean(c2[1:Nx-1, jj], c2[1:Nx-1, jj+1]) + hmean(c2[1:Nx-1, jj], c2[2:Nx, jj]) + hmean(c2[1:Nx-1, jj], c2[0:Nx-2, jj]) + hmean(c2[1:Nx-1, jj], c2[1:Nx-1, jj-1]) + c0[1:Nx-1, jj]
        #lower[mm(1, jj)-lower_offset: mm(Nx-1, jj)-lower_offset] = -0.5*(c2[1:Nx-1, jj] + c2[0:Nx-2, jj])
        lower[mm(1, jj)-lower_offset: mm(Nx-1, jj)-lower_offset] = -hmean(c2[1:Nx-1, jj], c2[0:Nx-2, jj])
        #lower2[mm(1, jj)-lower2_offset: mm(Nx-1, jj)-lower2_offset] = -0.5*(c2[1:Nx-1, jj] + c2[1:Nx-1, jj-1])
        lower2[mm(1, jj)-lower2_offset: mm(Nx-1, jj)-lower2_offset] = -hmean(c2[1:Nx-1, jj], c2[1:Nx-1, jj-1])

        # ii = Nx-1 right boundary
        upper2[mm(Nx-1, jj)] = -c2[Nx-1, jj]
        main[mm(Nx-1, jj)] = 4.0 * c2[Nx-1, jj] + c0[Nx-1, jj]
        lower[mm(Nx-1, jj)-lower_offset] = -2.0 * c2[Nx-1, jj]
        lower2[mm(Nx-1, jj)-lower2_offset] = -c2[Nx-1, jj]

        # rhs
        b[mm(0, jj):mm(Nx, jj)] = f[:, jj]

    """
    jj = Ny-1 (upper boundary)
    """

    # ii = 0
    upper[mm(0, Ny-1)] = -2.0 * c2[0, Ny-1]
    main[mm(0, Ny-1)] = 4.0 * c2[0, Ny-1] + c0[0, Ny-1]
    lower2[mm(0, Ny-1)-lower2_offset] = -2.0 * c2[0, Ny-1]

    # 1<= ii <= Nx-1
    upper[mm(1, Ny-1): mm(Nx-1, Ny-1)] = -c2[1:Nx-1, Ny-1]
    main[mm(1, Ny-1): mm(Nx-1, Ny-1)] = 4.0 * c2[1:Nx-1, Ny-1] + c0[1:Nx-1, Ny-1]
    lower[mm(1, Ny-1)-lower_offset: mm(Nx-1, Ny-1)-lower_offset] = -c2[1:Nx-1, Ny-1]
    lower2[mm(1, Ny-1)-lower2_offset: mm(Nx-1, Ny-1)-lower2_offset] = -2.0 * c2[1:Nx-1, Ny-1]

    # ii = Nx
    main[mm(Nx-1, Ny-1)] = 4.0 * c2[Nx-1, Ny-1] + c0[Nx-1, Ny-1]
    lower[mm(Nx-1, Ny-1)-lower_offset] = -2.0 * c2[Nx-1, Ny-1]
    lower2[mm(Nx-1, Ny-1)-lower2_offset] = -2.0 * c2[Nx-1, Ny-1]

    # rhs
    b[mm(0, Ny-1): mm(Nx, Ny-1)] = f[:, Ny-1]

    A = sp.diags(
        diagonals=[lower2, lower, main, upper, upper2],
        offsets=[-lower2_offset, -lower_offset, 0, lower_offset, lower2_offset],
        shape=(N, N),
        format='csr')

    # solve system
    if (solver_type is None):
        return A, b, None, None
    elif (solver_type == 'sparse'):
        u1d = sla.spsolve(A, b)
    elif (solver_type == 'dense'):
        u1d = la.solve(A.toarray(), b)

    # reshape 1d vector to 2d array
    u = np.reshape(u1d, [Nx, Ny], 'F')

    return u, A, b


def hmean(u,v):
    # sanity check
    if not(len(u) == len(v)):
        raise ValueError('u and v size mismatch')

    u0 = abs(u)<1e-16
    v0 = abs(v)<1e-16
    out = np.zeros(len(u)) + np.nan
    uv0 = u0|v0
    out[uv0] = 0
    out[~uv0] = 2/(1/u[~uv0]+1/v[~uv0])
    # sanity check
    if not (np.all(np.isfinite(out))):
        raise ValueError('internal error')

    return out
