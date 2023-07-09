import numpy as np
from codes.braiding_generators.fib_multi_qudits import (
    check_state,
    find_basis,
    braiding_generator
    )


def fib_seq(n):
    """
    Fibonacci sequence: U(n+2)=U(n+1)+U(n), U(0)=U(1)=1
    """
    u = [1, 1]
    for i in range(n // 2):
        u[0] += u[1]
        u[1] += u[0]

    return u[n % 2]


def test_check_state():
    """ """
    assert check_state({'qudits': [[1, 0], [1, 0]], 'roots': [0]})
    assert check_state({'qudits': [[0, 1], [1, 0]], 'roots': [1]})
    assert check_state({'qudits': [[1, 1], [1, 1]], 'roots': [1]})
    assert check_state({'qudits': [[0, 1, 0], [0, 1, 0]], 'roots': [0]})
    assert check_state({'qudits': [[0, 1, 0], [0, 1, 0]], 'roots': [0]})
    assert check_state({'qudits': [[1, 0, 1], [0, 1, 0]], 'roots': [1]})


def test_find_basis():
    """ """
    for _ in range(3):
        nb_anyons = np.random.randint(1, 3)
        qudit_len = np.random.randint(2, 5)
        assert (
            len(find_basis(nb_anyons, qudit_len))
            == fib_seq(nb_anyons*(qudit_len+1))
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
