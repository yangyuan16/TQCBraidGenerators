#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 20:57:05 2020

Created on 2022

@author: A. Tounsi

This script calculates the matrix representation of iToffoli braiding gate
approximated using controlled injection method following the procedure of
the paper.

Braiding generators are computed by fibonacci_2q6a.py
"""
import pickle
import numpy as np
from codes.braid_matrix_calculator import error_distance, get_matrix, leakage_error
from codes.transformer import uncouple, time_mirror, uncouple_all
import codes.braiding_generators.ising_multi_qudits as multi_q
from codes.cplot import cplot, scale


# Calculate sigmas of 3 qubits 12 anyons

try:
    with open("bin/basis-3q-12a-ising.pickle", "rb") as file:
        basis = pickle.load(file)

    with open("bin/SIG-3q-12a-ising.pickle", "rb") as file:
        SIG = pickle.load(file)

except FileNotFoundError:
    SIG = {}
    for index in range(11):
        n = index + 1
        SIG[n] = {}
        gen = multi_q.braiding_generator(n, 3, 3, show=False)

        SIG[n][1] = np.array(gen[0])
        SIG[n][-1] = np.array(np.linalg.inv(gen[0]))

    basis = gen[1]
    with open("bin/basis-3q-12a-ising.pickle", "wb") as file:
        pickle.dump(basis, file)
    with open("bin/SIG-3q-12a-ising.pickle", "wb") as file:
        pickle.dump(SIG, file)

R = {'sigma': [8, 7, 9, 8,
               6, 5, 7, 6, 6, 7, 5, 6, 6, 5, 7, 6, 6, 7, 5, 6,
               6, 5, 7, 6, 6, 7, 5, 6, 6, 5, 7, 6, 6, 7, 5, 6,
               6, 5, 7, 6, 6, 7, 5, 6, 6, 5, 7, 6, 6, 7, 5, 6,
               6, 5, 7, 6, 6, 7, 5, 6,
               6, 5, 7, 6], 
     'power': []}

for _ in range(4*16):
    R['power'].append(1)

I = {}
R_inv = time_mirror(R)
result_seq = R
result_seq['sigma'] = R['sigma'] + R_inv['sigma']
result_seq['power'] = R['power'] + R_inv['power']

# Calculate matrix representation of the braiding sequance
w = get_matrix(result_seq, sigma=SIG)

# Extract the matrix in computational space
result_gate = np.zeros([8, 8]) * (1 + 0j)
bases = []
row = 0
for bb, base in enumerate(basis):
    if (
        base["qudits"][0][2] == 0
        and base["qudits"][1][2] == 0
        and base["qudits"][2][2] == 0
    ):
        column = 0
        bases.append(base)
        for cc, case in enumerate(basis):

            if (
                case["qudits"][0][2] == 0
                and case["qudits"][1][2] == 0
                and case["qudits"][2][2] == 0
            ):
                result_gate[row, column] = w[bb, cc]
                column += 1
        row += 1

factor = 1 / np.sqrt(2)

real_gate = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1j],
        [0, 0, 0, 0, 0, 0, -1j, 0],
    ]
)

sigma_ = 0.01
cplot(result_gate, title="toffoli-l48", sigma=sigma_, show=True)
cplot(real_gate, title="toffoli-exact", sigma=sigma_, show=True)
scale(sigma=sigma_, show=False)

print("iToffoli using controlled-injection method")
print("--------------------------")
print("overall error =")
print(error_distance(result_gate, real_gate))
print("target's error")
print("[00]", error_distance(result_gate[0:2, 0:2], real_gate[0:2, 0:2]))
print("[01]", error_distance(result_gate[2:4, 2:4], real_gate[2:4, 2:4]))
print("[10]", error_distance(result_gate[4:6, 4:6], real_gate[4:6, 4:6]))
print("[11]", error_distance(result_gate[6:8, 6:8], real_gate[6:8, 6:8]))
print("leakage error")
print(leakage_error(result_gate))
print("--------------------------")
