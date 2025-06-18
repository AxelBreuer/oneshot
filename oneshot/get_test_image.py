# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:49:43 2025

@author: GD5264
"""

import numpy as np


def get_test_image(image_name):

    if (image_name == 'circle'):
        Nx = 60
        Ny = 50
        image = np.zeros([Nx, Ny])
        for ii in range(Nx):
            for jj in range(Ny):
                if (((ii-25)**2+(jj-20)**2) < 10**2):
                    image[ii, jj] = 1
        ismissing = np.zeros([Nx, Ny], bool)
        ismissing[25:45, 25:35] = True
        image[ismissing] = 0.0

    elif (image_name == 'square'):
        Nx = 64
        Ny = 64
        image = np.zeros([Nx, Ny])
        image[:Nx//2, :Ny//2] = 1.0
        ismissing = np.zeros([Nx, Ny], bool)
        ismissing[int(3*Nx/8): int(Nx*3/4), int(3*Ny/8): int(Ny*3/4)] = True
        image[ismissing] = 0.0

    elif (image_name == 'toy'):
        Nx = 5
        Ny = 5
        image = np.zeros([Nx, Ny])
        image[0,:] = 1.0
        image[1,1:] = 1.0
        image[2,2] = 1.0
        ismissing = np.zeros([Nx, Ny], bool)
        ismissing[1:4,1:4] = True
        image[ismissing] = 0.0

    else:
        raise ValueError('Do not know how to get image_name = ' + image_name)

    return image, ismissing
