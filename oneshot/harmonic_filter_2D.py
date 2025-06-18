# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:05:17 2022

@author: GD5264
"""

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def harmonic_filter_2D(ys, l2s, interp_p=False, verbose=True):

    func_name = 'harmonic_filter_2D()'

    # TODO : avoid using cvxpy optimizer, and use scipy.sparse.linalg solvers instead.

    # sanity check
    """
    if not(l2s.shape[0] == (ys.shape[0]-1)):
        raise ValueError('must have l2s.shape[0] == ys.shape[0]-1')
    if not(l2s.shape[1] == (ys.shape[1]-1)):
        raise ValueError('must have l2s.shape[1] == ys.shape[1]-1')
    """

    Nx, Ny = ys.shape

    isknown = np.isfinite(ys)

    # build the problem
    tv_ys = cp.Variable((Nx,Ny))

    diffx = tv_ys[1:, :Ny-1] - tv_ys[:Nx-1, :Ny-1]
    diffy = tv_ys[:Nx-1, 1:] - tv_ys[:Nx-1, :Ny-1]

    #diffx = tv_ys[1:, :] - tv_ys[:-1, :]
    #diffy = tv_ys[:, 1:] - tv_ys[:, :-1]

    if interp_p:
        obj = cp.Minimize(cp.sum_squares(cp.multiply(l2s, diffx)) + cp.sum_squares(cp.multiply(l2s, diffy)))
        # obj = cp.Minimize(cp.sum_squares(cp.multiply((l2s[1:,:]+l2s[:-1,:])/2, diffx)) + cp.sum_squares(cp.multiply((l2s[:,1:]+l2s[:,:-1])/2, diffy)))
        constraints = [tv_ys[isknown] == ys[isknown]]
        prob = cp.Problem(obj, constraints)
    else:
        obj = cp.Minimize(cp.sum_squares(tv_ys[isknown]-ys[isknown]) + cp.sum_squares(cp.multiply(l2s, diffx)) + cp.sum_squares(cp.multiply(l2s, diffy)))
        # obj = cp.Minimize(cp.sum_squares(cp.multiply((l2s[1:,:]+l2s[:-1,:])/2, diffx)) + cp.sum_squares(cp.multiply((l2s[:,1:]+l2s[:,:-1])/2, diffy)) + cp.sum_squares(tv_ys[isknown]-ys[isknown]))
        prob = cp.Problem(obj)

    # solve the problem
    result = prob.solve(verbose=False, solver='OSQP')#, max_iters=1000)
    status = prob.status

    if (verbose and not(status == 'optimal')):
        print(func_name + ': WARNING optimizer did not converge status = ' + prob.status)
        stop

    return tv_ys.value, obj.value
