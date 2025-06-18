# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:05:17 2022

@author: GD5264
"""

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def harmonic_filter(ys, l2s, interp_p=False, verbose=True):

    func_name = 'harmonic_filter()'

    # TODO : avoid using cvxpy optimizer, and use scipy.sparse.linalg solvers instead.

    # sanity check
    if not(len(l2s) == (len(ys)-1)):
        raise ValueError('must have len(l1) == len(ys)-1')

    T = len(ys)

    ismissing = ~np.isfinite(ys)

    D1 = sp.diags(np.ones(T-1), 1) - sp.diags(np.ones(T))
    D1 = D1.tocsc()
    D1 = D1[:T-1, :]

    # build the problem
    tv_ys = cp.Variable(T)
    if interp_p:
        obj = cp.Minimize(cp.sum_squares(cp.multiply(l2s, D1@tv_ys)))
        constraints = [(ys[~ismissing] == tv_ys[~ismissing])]
        prob = cp.Problem(obj, constraints)
    else:
        obj = cp.Minimize(cp.sum_squares(ys[~ismissing] - tv_ys[~ismissing]) + cp.sum_squares(cp.multiply(l2s, D1@tv_ys)))
        prob = cp.Problem(obj)

    # solve the problem
    result = prob.solve(verbose=True, solver='ECOS')
    status = prob.status

    if (verbose and not(status == 'optimal')):
        print(func_name + ': WARNING optimizer did not converge status = ' + prob.status)

    return tv_ys.value, obj.value
