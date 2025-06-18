# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 11:04:50 2025

@author: GD5264
"""

import matplotlib.pyplot as plt


def pcolor_xy(arr, title, xlabel="x", ylabel="y", bad_color="white"):

    plt.figure()
    plt.pcolor(arr.T)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=bad_color)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

