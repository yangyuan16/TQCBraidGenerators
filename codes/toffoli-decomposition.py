#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 20:57:05 2020

Created on 2022

@author: A. Tounsi

This script calculates the matrix representation of iToffoli braiding gate
approximated using the decomposition method following the procedure of
the paper.

Braiding generators are computed by fibonacci_2q6a.py
"""
import pickle
import numpy as np
from braid_matrix_calculator import error_distance, get_matrix, leakage_error
from transformer import uncouple, time_mirror
import braiding_generators.fib_multi_qudits as multi_q
from cplot import cplot, scale


# Calculate sigmas of 3 qubits 12 anyons

try:
    with open("bin/basis-3q-12a.pickle", "rb") as file:
        basis = pickle.load(file)

    with open("bin/SIG-3q-12a.pickle", "rb") as file:
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
    with open("bin/basis-3q-12a.pickle", "wb") as file:
        pickle.dump(basis, file)
    with open("bin/SIG-3q-12a.pickle", "wb") as file:
        pickle.dump(SIG, file)

# sub gates

# create C-sqrt-NOTs

# id 48 (Bonesteel paper)
I_seq = uncouple(
    [2, -2, 2, 2, -2, 2, 4, -2, -4, -4, -2, -2, 2, 4, 2, -4, -2, 2],
    s0=2,
    init_strand=3,
    final_strand=1,
    rank_increment=2,
)

# id 31 l 48 (Toshiba)
I_seq = uncouple(
    [-2, -4, -2, 2, -2, -4, -4, 2, -2, -2, 2, -2, -4, -2, 2, -2, -2, -2, -2],
    s0=1,
    init_strand=3,
    final_strand=-1,
    rank_increment=2,
)

# L = 48 Toshiba
sX_seq = uncouple(
    [2, 4, 4, -2, 2, 4, 2, 2, -2, 4, 2, -2, 2, -4, -2, 4, 2],
    s0=1,
    init_strand=-3,
    final_strand=-3,
    rank_increment=0,
)

sX_seq_dagger = time_mirror(sX_seq)


invI_seq = time_mirror(I_seq)

# Controlled-sqrtNOT
csqrtx_seq = {}
csqrtx_seq["sigma"] = (
    I_seq["sigma"]
    + sX_seq["sigma"]
    + invI_seq["sigma"]
)

csqrtx_seq["power"] = (
    I_seq["power"]
    + sX_seq["power"]
    + invI_seq["power"])

csqrtxdagger_seq = {}
csqrtxdagger_seq["sigma"] = (
    I_seq["sigma"]
    + sX_seq_dagger["sigma"]
    + invI_seq["sigma"]
)

csqrtxdagger_seq["power"] = (
    I_seq["power"]
    + sX_seq_dagger["power"]
    + invI_seq["power"]
)

# create CNOTs

# x33 l 48 (Toshiba ) optimal
X_seq = uncouple(
    [2, 2, 4, 2, -4, 2, -4, -2, 4, -2, -2, 2, 2, -2, -2, -2, -2, -2, 2],
    s0=2,
    init_strand=3,
    final_strand=3,
    rank_increment=4,
)


I_seq = uncouple(
    [2, -2, 2, 2, -2, 2, 4, -2, -4, -4, -2, -2, 2, 4, 2, -4, -2, 2],
    s0=2,
    init_strand=3,
    final_strand=1,
    rank_increment=6,
)

# I_seq = uncouple(
#     [-2, -4, -2, 2, -2, -4, -4, 2, -2, -2, 2, -2, -4, -2, 2, -2, -2, -2, -2],
#     s0=1, init_strand=3, final_strand=-1, rank_increment=6
# )

invX_seq = time_mirror(X_seq)
invI_seq = time_mirror(I_seq)

cnot_seq = {}
cnot_seq["sigma"] = I_seq["sigma"] + X_seq["sigma"] + invI_seq["sigma"]
cnot_seq["power"] = I_seq["power"] + X_seq["power"] + invI_seq["power"]

inv_cnot_seq = {}
inv_cnot_seq["sigma"] = I_seq["sigma"] + invX_seq["sigma"] + invI_seq["sigma"]
inv_cnot_seq["power"] = I_seq["power"] + invX_seq["power"] + invI_seq["power"]

# create SWAPs
swap_seq = {}
swap_seq["sigma"] = [8, 7, 6, 5, 9, 8, 7, 6, 10, 9, 8, 7, 11, 10, 9, 8]
swap_seq["power"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

swap_seq_inv = {}
swap_seq_inv["sigma"] = [8, 7, 6, 5, 9, 8, 7, 6,
                         10, 9, 8, 7, 11, 10, 9, 8][::-1]
swap_seq_inv["power"] = [-1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
swap_seq_inv = time_mirror(swap_seq)

# Tofolli using decomposition method

toffoli_seq = {}

toffoli_seq["sigma"] = (
    csqrtx_seq["sigma"]  # controlled-sqrt-NOT
    + cnot_seq["sigma"]  # CNOT
    + csqrtxdagger_seq["sigma"]  # controlled-sqrt-NOT^-1
    + inv_cnot_seq["sigma"]  # CNOT^-1
    + swap_seq["sigma"]  # SWAP
    + csqrtx_seq["sigma"]  # controlled-sqrt-NOT
    + swap_seq_inv["sigma"]  # SWAP
)

toffoli_seq["power"] = (
    csqrtx_seq["power"]
    + cnot_seq["power"]
    + csqrtxdagger_seq["power"]
    + inv_cnot_seq["power"]
    + swap_seq["power"]
    + csqrtx_seq["power"]
    + swap_seq_inv["power"]
)


# Calculate matrix representation of C-NOT braiding sequance
w = get_matrix(toffoli_seq, sigma=SIG)

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
cplot(result_gate, title="toffoli-deco", sigma=sigma_, show=False)
cplot(real_gate, title="toffoli-deco-exact", sigma=sigma_, show=False)
scale(sigma=sigma_, show=False)

print("iToffoli using decomposition method")
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
