#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:45:49 2023

@author: thinkpo
"""
import numpy as np
from math import isclose
from braid_matrix_calculator import error_distance


I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

def random_unitary():
        
    a = np.random.rand()
    b = np.random.rand() * (1 - a)
    c = np.random.rand() * (1 - a - b)
    d = 1 * (1 - a - b - c)
    
    return np.sqrt(a)*I -1j*( np.sqrt(b)*X + np.sqrt(c)*Y + np.sqrt(d)*Z )

def random_q():
        
    a = np.random.rand()
    b = np.random.rand() * (1 - a)
    c = np.random.rand() * (1 - a - b)
    d = 1 * (1 - a - b - c)
    
    return np.sqrt(np.array(a, b, c, d))

unitary = lambda a: a[0]*I - 1j*a[1]*X - 1j*a[2]*Y - 1j*a[3]*Z

c = 0
var = []
for _ in range(100):
    
    u = random_unitary()
    v = random_unitary()
    
    #print(error_distance(u, v))
    #print(error_distance(u@u, v@v))
    #print('----')
    
    var.append(error_distance(u@u, v@v) - 2*error_distance(u, v))
    if var[-1] < 0:
    #if isclose(var[-1], 0, rel_tol=1e-1):
        c += 1
        print(u, v)
        print(error_distance(u, v))
        print('-----')

print(c/100)

import matplotlib.pyplot as plt

plt.hist(var)

