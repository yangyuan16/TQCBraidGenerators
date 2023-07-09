import numpy as np
from codes.braiding_generators.ising_multi_qudits import (
    check_state,
    find_basis,
    braiding_generator
    )

import codes.braiding_generators.ising_qudit as ising


def test_check_state():
    """ """
    assert check_state({'qudits': [[0, 1, 0], [0, 1, 0]], 'roots': [0]})
    assert check_state({'qudits': [[2, 1, 0], [2, 1, 2]], 'roots': [2]})
    assert not check_state({'qudits': [[0, 1, 0], [0, 1, 0]], 'roots': [2]})
    assert not check_state({'qudits': [[2, 1, 2], [0, 1, 2]], 'roots': [1]})


def test_find_basis():
    """ """
    for _ in range(3):
        nb_anyons = np.random.randint(1, 3)
        qudit_len = np.random.randint(2, 5)
        assert (
            len(find_basis(nb_anyons, qudit_len))
            == 2**(nb_anyons*(qudit_len+1)//2)
        )


def test_find_basis_consistency():
    """ """
    for nb_anyon in range(3, 6):
        assert len(find_basis(1, nb_anyon-1)) == len(
            ising.find_basis(nb_anyon)
            )


def test_braiding_generator():
    """ """
    nb_anyons_per_qubit = 4
    nb_qubits = 2
    nb_anyons = nb_anyons_per_qubit * nb_qubits
    sigmas = []
    for index in range(1, nb_anyons):
        sigmas.append(np.array(braiding_generator(index,
                                                  nb_qubits,
                                                  nb_anyons_per_qubit-1,
                                                  show=False)[0]))
        np.testing.assert_allclose(sigmas[-1] @ sigmas[-1].conjugate().T,
                                   np.eye(sigmas[-1].shape[0]),
                                   rtol=1e-5, atol=1e-5)

    for index in range(1, nb_anyons-2):
        np.testing.assert_allclose(
            sigmas[index] @ sigmas[index+1] @ sigmas[index],
            sigmas[index+1] @ sigmas[index] @ sigmas[index+1],
            rtol=1e-5, atol=1e-5
        )
