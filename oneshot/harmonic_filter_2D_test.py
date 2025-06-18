# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:05:17 2022

@author: GD5264
"""

"""
build example to inpaint
"""

import matplotlib.pyplot as plt
import numpy as np

from harmonic_filter_2D import harmonic_filter_2D

Nx = 64
Ny = 64
example = 'circle'

if (example == 'circle'):

    f = np.zeros([Nx, Ny])
    for ii in range(Nx+1):
        for jj in range(Ny+1):
            if (((ii-Nx//2)**2+(jj-Ny//2)**2) < (min(Nx,Ny)**2)/16):
                f[ii, jj] = 1.0
    ismissing = np.zeros([Nx, Ny], bool)
    ismissing[Nx//2: int(Nx*7/8)+1, Ny//2: int(Ny*7/8)+1] = True
    f[ismissing] = np.nan

elif (example == 'square'):

    f = np.zeros([Nx, Ny])
    f[:Nx//2, :Ny//2] = 1.0
    ismissing = np.zeros([Nx, Ny], bool)
    ismissing[int(3*Nx/8): int(Nx*3/4), int(3*Ny/8): int(Ny*3/4)] = True
    f[ismissing] = np.nan

elif (example == 'toy'):
    Nx = 5
    Ny = 5
    f = np.zeros([Nx, Ny])
    f[0,:] = 1.0
    f[1,1:] = 1.0
    f[2,2] = 1.0
    ismissing = np.zeros([Nx, Ny], bool)
    ismissing[1:4,1:4] = True
    f[ismissing] = np.nan

else:
    raise ValueError('Do not know how to intialize when example = ' + example)


interp_p = False

plt.close('all')

c2s = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 1, 1.1, 5, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
objs = np.zeros(len(c2s))
subobj_tracks = np.zeros(len(c2s))
subobj_diffs = np.zeros(len(c2s))
for cc in range(len(c2s)):
    c2 = c2s[cc]
    l2s = c2 * np.ones((Nx-1, Ny-1))
    #l2s = c2 * np.ones((Nx, Ny))
    harmonic_f, obj = harmonic_filter_2D(f, l2s, interp_p=interp_p)
    subobj_track = np.sum((f[np.isfinite(f)] - harmonic_f[np.isfinite(f)])**2)
    subobj_diff = np.sum((harmonic_f[1:,1:-1]-harmonic_f[1:,1:-1])**2) + np.sum((harmonic_f[1:-1,1:]-harmonic_f[1:-1,:-1])**2)
    print(c2, obj, subobj_track, subobj_diff)
    objs[cc] = obj
    subobj_tracks[cc] = subobj_track
    subobj_diffs[cc] = subobj_diff

plt.figure()
plt.loglog(c2s, objs, 'o-')
plt.loglog(c2s, subobj_tracks, 'o-')
plt.loglog(c2s, subobj_diffs, 'o-')
plt.legend(['obj', 'subobj_track', 'subobj_diff'])

c2 = 1.0
l2s = c2 * np.ones((Nx-1, Ny-1))
#l2s = c2 * np.ones((Nx, Ny))
harmonic_f, obj = harmonic_filter_2D(f, l2s, interp_p=interp_p)

plt.figure()
plt.pcolor(f)

plt.figure()
plt.pcolor(harmonic_f)
plt.title('interp_p = ' + str(interp_p) + ' | c2 = ' + str(c2))

objs_d = np.zeros([Nx-1, Ny-1]) + np.nan
objs_n = np.zeros([Nx-1, Ny-1]) + np.nan
objs = np.zeros([Nx-1, Ny-1]) + np.nan
objs_d = np.zeros([Nx-1, Ny-1]) + np.nan
objs_n = np.zeros([Nx-1, Ny-1]) + np.nan

# objs_d = np.zeros([Nx, Ny]) + np.nan
# objs_n = np.zeros([Nx, Ny]) + np.nan
# objs = np.zeros([Nx, Ny]) + np.nan
# objs_d = np.zeros([Nx, Ny]) + np.nan
# objs_n = np.zeros([Nx, Ny]) + np.nan

harmonic_f_d0, obj_d0 = harmonic_filter_2D(f, l2s, interp_p=True)
harmonic_f_n0, obj_n0 = harmonic_filter_2D(f, l2s, interp_p=False)
harmonic_f_dn0 = harmonic_f_d0 - harmonic_f_n0
harmonic_f_dn0[~ismissing] = 0.0
harmonic_f_dn02 = harmonic_f_dn0**2
obj0 = np.sum(harmonic_f_dn02)

idx = 0
for ii in range(Nx-1):
    for jj in range(Ny-1):
        if not(ismissing[ii, jj]):
            continue
        l2s_tmp = l2s.copy()
        l2s_tmp[ii, jj] = c2*0.1
        harmonic_f_d, obj_d = harmonic_filter_2D(f, l2s_tmp, interp_p=True)
        harmonic_f_n, obj_n = harmonic_filter_2D(f, l2s_tmp, interp_p=False)
        objs_d[ii, jj] = obj_d
        objs_n[ii, jj] = obj_n
        harmonic_f_dn = harmonic_f_d - harmonic_f_n
        harmonic_f_dn[~ismissing] = 0.0
        harmonic_f_dn2 = harmonic_f_dn**2
        objs[ii, jj] = np.sum(harmonic_f_dn2) - obj0
        idx += 1
        print(idx/ismissing.sum())

plt.figure()
plt.pcolor(objs)
plt.colorbar()

plt.figure()
plt.plot(np.sort(objs.ravel()), 'o-')

objs_thres = -0.05e-4
plt.figure()
plt.pcolor(objs<objs_thres)
plt.colorbar()

l2s_tmp = l2s.copy()
l2s_tmp[objs<objs_thres] = c2*0.1

harmonic_f_d, obj_d = harmonic_filter_2D(f, l2s_tmp, interp_p=True)
harmonic_f_n, obj_n = harmonic_filter_2D(f, l2s_tmp, interp_p=False)

plt.figure()
plt.pcolor(harmonic_f_d)
plt.colorbar()

plt.figure()
plt.pcolor(harmonic_f_n)
plt.colorbar()
