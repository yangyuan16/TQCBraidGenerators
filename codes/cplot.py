#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 21:54:22 2020

@author: abduhu

Complex unitary matrix plotting
********


"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, cos, atan, exp
from colorsys import hls_to_rgb


def cplot(unitary: 'np.ndarray', title="", sigma=0.5, show=True, ticks=None):
    """
    Plots complex matrix using chromatic values.
    """
    dims = unitary.shape
    img = []
    for r, row in enumerate(unitary):
        img.append([])
        for c in row:
            y = c.imag
            x = c.real

            if x == 0:
                if y > 0:
                    theta = pi / 2
                else:
                    theta = -pi / 2
            else:
                theta = atan(y / x)
            if x < 0:
                theta += pi

            hue = theta / (2 * pi)
            rad = sqrt(x**2 + y**2)
            lum = 0.5 + 0.5 * exp(-rad / sigma)
            if rad > 2:
                sat = 0
            else:
                sat = cos(pi * rad / 2) * 0.5 + 0.5
            img[-1].append(hls_to_rgb(hue, lum, sat))
    
    fig, axs = plt.subplots(nrows=1, figsize=[4, 4])
    
    if ticks == None:
        plt.xticks([i for i in range(dims[0])])
        plt.yticks([dims[1] - 1 - i for i in range(dims[1])])
    else:
        plt.xticks([i for i in range(dims[0])],
                   labels=ticks)
        plt.yticks([dims[1] - 1 - i for i in range(dims[1])],
                   labels=ticks[::-1])
        
    plt.grid(linewidth=0.1)
    plt.imshow(img)  # , extent=(1, dims[0], 1, dims[1]))
    axs.xaxis.set_ticks_position("top")
    plt.axis()
    plt.savefig(f"images/{title}.png", dpi=500, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def scale(sigma=0.5, title="scale", show=True):
    """
    Plot the scaling spectrum of the complex plane [-1, 1, -i, i]
    """
    img = []
    sc = 300
    for r in range(-sc, sc):
        img.append([])
        for c in range(-sc, sc):
            y = r / sc
            x = c / sc
            if x == 0:
                if y > 0:
                    theta = pi / 2
                else:
                    theta = 3 * pi / 2
            else:
                theta = atan(y / x)
            if x < 0:
                theta += pi
            if theta < 0:
                theta += 2 * pi

            hue = theta / (2 * pi)
            rad = sqrt(x**2 + y**2)
            lum = 0.5 + 0.5 * exp(-rad / sigma)
            if rad > 2:
                sat = 0
            else:
                sat = cos(pi * rad / 2) * 0.5 + 0.5
            img[-1].append(hls_to_rgb(hue, lum, sat))

    fig, axs = plt.subplots(nrows=1, figsize=[4, 4])
    plt.imshow(img[::-1], extent=(-1, 1, -1, 1))
    axs.set_xticks([-1, -0.5, 0.5, 1])
    #plt.xticks([-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
    #           rotation = 90)
    axs.xaxis.set_ticks_position("top")
    axs.xaxis.set_label_position('top')
    plt.tick_params(axis='both', labelsize=10)
    plt.tick_params(axis='x',)# rotation=90)
    axs.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs.set_yticks([-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1])
    plt.xlabel("Re")
    plt.ylabel("Img")
    plt.tight_layout()
    plt.grid(linewidth=0.1)
    #plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=12)
    plt.savefig(f"images/{title}.png", dpi=500, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
