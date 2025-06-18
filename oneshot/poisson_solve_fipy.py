# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:38:25 2021

@authors: Axel Breuer and Didier Auroux
"""

import numpy as np


from fipy import CellVariable, Grid2D, DiffusionTerm, ImplicitSourceTerm
from fipy import numerix


def poisson_solve_fipy(c0, c2, f, pin_mask):

    """
    Return u, solution of the second order elliptic PDE

        - div[c2(x,y) * grad[u(x,y)]] + c0(x,y) u(x,y) = f(x,y)

    on a rectangular domain with Neumann boundary conditions.

    When c2*c0>0, the PDE is called 'Screened Poisson equation'
    When c2*c0<0, the PDE is called 'Helmholtz equation'

    This PDE is discretized by finite volume using 'fipy'.

    pin_mask argument is used to enforce Dirichlet conditions in 'fipy'
    """

    func_name = "poisson_solve_fipy()"

    # sanity check
    if not(c0.shape == f.shape):
        raise ValueError("c0 and f must have same shape")
    if not(c2.shape == f.shape):
        raise ValueError("c2 and f must have same shape")

    Nx, Ny = c0.shape

    mesh = Grid2D(dx=1.0, dy=1.0, nx=Nx, ny=Ny)

    # Coordinates
    x, y = mesh.cellCenters

    # Flatten values
    # c2_flat are c2 values attached to (x[0], y[0]), (x[1],y[0]), (x[2], y[0]), ...
    # hence c2_flat = [c2[:, 0], c2[:, 1], ....]
    # hence ravel with order='F'
    c2_flat = c2.ravel(order='F')
    c0_flat = c0.ravel(order='F')
    f_flat  = f.ravel(order='F')

    # Wrap them as CellVariables
    c2_cv = CellVariable(mesh=mesh, value=c2_flat)
    c0_cv = CellVariable(mesh=mesh, value=c0_flat)
    f_cv = CellVariable(mesh=mesh, value=f_flat)

    # Define solution variable
    u = CellVariable(mesh=mesh, name="u", value=0.0)

    # Neumann BCs: nothing to do (default zero-flux)

    # Define the PDE: -div(c2 * grad(u)) + c0 * u = f
    eq = - DiffusionTerm(coeff=c2_cv) + ImplicitSourceTerm(coeff=c0_cv) == f_cv

    # Apply Dirichlet constraints on known region only
    if not (pin_mask is None):
        pin_indices = np.where(pin_mask)[0]
        cell_indices = numerix.arange(mesh.numberOfCells)

        for idx in pin_indices:
            u.constrain(f_flat[idx], where=(cell_indices == idx))

    # Solve
    eq.solve(var=u)

    uu = u.value.reshape((Nx, Ny), order='F')
    grad_u_x = np.array(u.grad[0]).reshape((Nx, Ny), order='F')
    grad_u_y = np.array(u.grad[1]).reshape((Nx, Ny), order='F')

    return uu, grad_u_x, grad_u_y
