# -*- coding: utf-8 -*-
"""
Created on 2022

@author: A. Tounsi

This script calculates the matrix representation of M(iX,iX) braiding gate
approximated using controlled injection method following the procedure of
the paper.

Braiding generators are computed by fibonacci_2q6a.py
"""
import pickle
import numpy as np
from braid_matrix_calculator import error_distance, get_matrix, leakage_error
from transformer import uncouple, time_mirror, uncouple_all
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


# Pre-Gates for Toffoli

xup_seq = uncouple(
    [2, 2, 4, 2, -4, 2, -4, -2, 4, -2, -2, 2, 2, -2, -2, -2, -2, -2, 2],
    s0=2,
    init_strand=3,
    final_strand=3,
    rank_increment=8,
    n_coupled_strands=1,
)

inv_xup_seq = time_mirror(xup_seq)

xmid_seq = uncouple(
    [2, 2, 4, 2, -4, 2, -4, -2, 4, -2, -2, 2, 2, -2, -2, -2, -2, -2, 2],
    s0=2,
    init_strand=3,
    final_strand=3,
    rank_increment=4,
    n_coupled_strands=1,
)

inv_xmid_seq = time_mirror(xmid_seq)

# R and I gates

x31_seq = uncouple_all(
    [-2, 2, 2, -4, 4, -4, 2, -2, -2, 2, -4, 2, -4, -2, 4, -2, -2],
    s0=1,
    init_strand=-3,
    final_strand=1,
    rank_increment=4,
)
# l 48 (Toshiba 2 cores) optimal too
# x31_seq = uncouple_all(
#     [2, -2, -2, 2, 2, -4, 2, 2, 2, -4, 2, -2, 2, -2, 4, -2, -4, 2, -2],
#     s0=1, init_strand=3, final_strand=1,
#     rank_increment=4
# )

inv_x31_seq = time_mirror(x31_seq)


id31up_seq = uncouple_all(
    [-2, -4, -2, 2, -2, -4, -4, 2, -2, -2, 2, -2, -4, -2, 2, -2, -2, -2, -2],
    s0=1,
    init_strand=3,
    final_strand=-1,
    rank_increment=4,
)

# another otption
# id31up_seq = uncouple_all(
#     [ 2, 4, 2, 4, 2, -2, 2, 4, 2, 4, -2, 2, -2, 2, 4, 2, -2, 2,  ],
#     s0=1, init_strand=3, final_strand=-1,
#     rank_increment=4
# )

inv_id31up_seq = time_mirror(id31up_seq)

# id 48 (Bonesteel paper)
# id31_seq = uncouple(
#     [2, -2, 2, 2, -2, 2, 4, -2, -4, -4, -2, -2, 2, 4, 2, -4, -2, 2],
#     s0=2, init_strand=3, final_strand=1,
#     rank_increment=2, n_coupled_strands=4
# )

# id 48 (489 min with Toshiba 2 cores)
id31_seq = uncouple(
    [-2, -4, -2, 2, -2, -4, -4, 2, -2, -2, 2, -2, -4, -2, 2, -2, -2, -2, -2],
    s0=1,
    init_strand=3,
    final_strand=-1,
    rank_increment=2,
    n_coupled_strands=4,
)

inv_id31_seq = time_mirror(id31_seq)

# Target gate

# x33 l 48 (Toshiba ) optimal
x33_seq = uncouple(
    [2, 2, 4, 2, -4, 2, -4, -2, 4, -2, -2, 2, 2, -2, -2, -2, -2, -2, 2],
    s0=2,
    init_strand=3,
    final_strand=3,
    rank_increment=0,
    n_coupled_strands=4,
)

# x33_seq = uncouple(
#     [4, -2, -4, 4, 2, 4, -2, 2, 2, 4, -2, -2, -2, -4, -4, 2  ],
#     s0=2, init_strand=-3, final_strand=-3,
#     rank_increment=0, n_coupled_strands=4
# )


inv_x33_seq = time_mirror(x33_seq)

# sx33_seq = uncouple(
#     [-2, 4, -2, -4, 2, -2, 4, -2, 2, -4, 2, -2, 2, 2, -2, -2,-4, -2],
#     s0=2, init_strand=-3, final_strand=3,
#     rank_increment=0, n_coupled_strands=1
# )

sx33_seq = uncouple(
    [2, 2, 4, 2, -4, 2, -4, -2, 4, -2, -2, 2, 2, -2, -2, -2, -2, -2, 2],
    s0=2,
    init_strand=3,
    final_strand=3,
    rank_increment=0,
    n_coupled_strands=1,
)

# sx33_seq = uncouple(
#     [4, -2, -4, 4, 2, 4, -2, 2, 2, 4, -2, -2, -2, -4, -4, 2  ],
#     s0=2, init_strand=-3, final_strand=-3,
#     rank_increment=0, n_coupled_strands=1
# )

inv_sx33_seq = time_mirror(sx33_seq)

# M(iX, iX)

andd = {}

andd["sigma"] = (
    x31_seq["sigma"]  # R=iX
    + id31_seq["sigma"]  # I
    + x33_seq["sigma"]  # S=iX
    + inv_id31_seq["sigma"]  # I^-1
    + inv_x31_seq["sigma"]  # R^-1=iX^-1
)

andd["power"] = (
    x31_seq["power"]  # R=iX
    + id31_seq["power"]  # I
    + x33_seq["power"]  # S=iX
    + inv_id31_seq["power"]  # I^-1
    + inv_x31_seq["power"]  # R^-1=iX^-1)
)

result_seq = andd


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
        [0, 0, 0, -1j, 0, 0, 0, 0],
        [0, 0, -1j, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1j, 0, 0],
        [0, 0, 0, 0, -1j, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1j],
        [0, 0, 0, 0, 0, 0, -1j, 0],
    ]
)

sigma_ = 0.01
cplot(result_gate, title="and-l48", sigma=sigma_, show=False)
cplot(real_gate, title="and-exact", sigma=sigma_, show=False)
scale(sigma=sigma_, show=False)

print("M(NOT, NOT) using controlled-injection method")
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
