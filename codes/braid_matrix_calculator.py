#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 20:23:40 2020

@author: abduhu
"""
import numpy as np
from copy import deepcopy
from typing import Dict


def get_global_phase(unitary: "np.ndarray") -> complex:
    """calculates the global phase of a unitary"""
    return pow(np.cfloat(np.linalg.det(unitary)), 1 / unitary.shape[0])


def error_distance(unitary_1: "np.ndarray", unitary_2: "np.ndarray") -> float:
    """Spectral distance metric between SU(n) matrices"""

    global_phase_1 = get_global_phase(unitary_1)
    global_phase_2 = get_global_phase(unitary_2)

    error = 2
    for sign in [-1, 1]:
        diff = unitary_1 / global_phase_1 + sign * unitary_2 / global_phase_2
        distance = abs(
            (max(np.linalg.eig(diff.conjugate().T @ diff)[0])) ** (1 / 2)
            )
        if distance < error:
            error = deepcopy(distance)
    return error


def leakage_error(unitary: "np.ndarray") -> float:
    """Leakage error using spectral definition"""

    return 1 - abs((
        min(np.linalg.eig(unitary.conjugate().T @ unitary)[0])) ** (1 / 2))


EX_SEQ = {
    "sigma": [1, 2, 1, 1],
    "power": [1, -1, 1, -1],
}  # chronology from left to right


PHI = (1.0 + np.sqrt(5.0)) / 2.0

SIG_1 = np.array(
    [[np.exp(-(4 / 5) * np.pi * 1j), 0], [0, np.exp((3 / 5) * np.pi * 1j)]]
)
SIG_2 = np.array(
    [
        [
            np.exp((4 / 5) * np.pi * 1j) / PHI,
            np.exp(-(3 / 5) * np.pi * 1j) / np.sqrt(PHI),
        ],
        [np.exp(-(3 / 5) * np.pi * 1j) / np.sqrt(PHI), -1.0 / PHI],
    ]
)

EX_SIGMA = {
    1: {1: SIG_1, -1: np.linalg.inv(SIG_1)},
    2: {1: SIG_2, -1: np.linalg.inv(SIG_2)},
}


def get_matrix(seq: Dict, sigma=0) -> "np.ndarray":
    """
    Calculates the matrix representation of given braid seq
    Example:
        print(EX_SEQ)
    Inputs:
        seq: dict:
            chronology from left to right
        sigma: dict
    Returns:
        numpy.array
    """
    if sigma == 0:
        sigma = EX_SIGMA

    result = np.eye(sigma[1][1].shape[0])

    for _, index in enumerate(seq["sigma"]):
        power = seq["power"][_]
        result = sigma[index][power] @ result

    return result
