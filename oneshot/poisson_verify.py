# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:36:29 2021

@author:Axel Breuer (with guidance by Didier Auroux)
"""

import numpy as np


def poisson_verify(c0, c2, f, u):

    """
    return residues such that

        residues(x,y) := - c2(x,y) Laplacian u(x,y) + c0(x,y) u(x,y) - f(x,y)

    If u(x,y) is the solution of the PDE

        - c2(x,y) Delta u(x,y) + c0(x,y) u(x,y) = f(x,y)

        with Neumann boundary condtions

    then residues(x,y) should be 0.0 (or very close to 0.0)

    To calculate the solution of the PDE, you can use poisson_solve()
    """

    Nx, Ny = c0.shape

    uu = np.zeros([Nx+2, Ny+2])

    # interior points
    uu[1:-1, 1:-1] = u

    # boundary points
    """
    uu[0, :] = uu[2, :]
    uu[-1, :] = uu[-3, :]
    uu[:, 0] = uu[:, 2]
    uu[:, -1] = uu[:, -3]
    """

    uu[0, :] = uu[1, :]
    uu[-1, :] = uu[-2, :]
    uu[:, 0] = uu[:, 1]
    uu[:, -1] = uu[:, -2]

    d2_u = uu[:-2, 1:-1] + uu[2:, 1:-1] + uu[1:-1, :-2] + uu[1:-1, 2:] - 4*uu[1:-1, 1:-1]

    residues = - c2 * d2_u + c0 * u - f

    return residues
