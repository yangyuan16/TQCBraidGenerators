import numpy as np
from codes.braid_matrix_calculator import (
    error_distance,
    leakage_error,
    get_matrix
    )


PAULI = [np.array([[1, 0], [0, 1]]) * (1 + 0j),
         np.array([[1, 0], [0, -1]]) * (-1j),
         np.array([[0, 1], [1, 0]]) * (-1j),
         np.array([[0, 1j], [-1j, 0]]) * (-1j)]


def random_special_unitary():
    """ genrate random SU(2) unitary matrix """
    coef = [0, 0, 0, 0]
    coef[0] = np.random.rand()
    coef[1] = np.random.rand() * np.sqrt(1 - coef[0]**2)
    coef[2] = np.random.rand() * np.sqrt(1 - coef[0]**2 - coef[1]**2)
    coef[3] = np.sqrt(1 - coef[0]**2 - coef[1]**2 - coef[2]**2)
    unitary = np.zeros([2, 2]) + 0j
    for __ in range(4):
        unitary += coef[__] * PAULI[__]
    return unitary


def test_error_distance():
    """ """
    unitary_1 = np.array([[1, 0], [0, 1]])
    unitary_2 = np.array([[1j, 0], [0, 1j]])
    unitary_3 = np.array([[0, 1], [1, 0]])

    assert np.isclose(error_distance(unitary_1, unitary_2), 0,
                      rtol=1e-5, atol=1e-5)
    assert np.isclose(error_distance(unitary_1, unitary_3), 1.414213562373095,
                      rtol=1e-5, atol=1e-5)
    assert np.isclose(error_distance(unitary_3, unitary_3 * (1+1j)/np.sqrt(2)),
                      0, rtol=1e-5, atol=1e-5)

    # metric properties
    for _ in range(3):

        unitary = [random_special_unitary(),
                   random_special_unitary(),
                   random_special_unitary()]

        for __ in range(3):
            # d(U, U) = 0
            assert np.isclose(error_distance(unitary[__], unitary[__]),
                              0, rtol=1e-5, atol=1e-5)
            assert np.isclose(error_distance(unitary[__], -unitary[__]),
                              0, rtol=1e-5, atol=1e-5)

        # Trigonometric relation
        # d(U, W) <= d(U, V) + d(V, W)
        assert (
            error_distance(unitary[0], unitary[1])
            + error_distance(unitary[1], unitary[2])
            >= error_distance(unitary[0], unitary[2])
        )


def test_leakage_error():
    """ """
    for _ in range(3):
        unitary = random_special_unitary()
        assert np.isclose(leakage_error(unitary), 0, rtol=1e-10, atol=1e-10)

    for dim in range(2, 5):
        unitary = np.eye(dim)
        unitary[0][dim-1] = 0.1
        unitary[dim-1][0] = 0.1
        assert np.isclose(leakage_error(unitary), 0.1, rtol=1e-10, atol=1e-10)


def test_get_matrix():
    """ """
    seq = {"sigma": [1, 1, 1, 1, 1],
           "power": [1, 1, 1, 1, 1]}
    np.testing.assert_allclose(get_matrix(seq),
                               np.array([[1, 0], [0, -1]]),
                               rtol=1e-10, atol=1e-10)
    seq = {"sigma": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           "power": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    np.testing.assert_allclose(get_matrix(seq),
                               np.array([[1, 0], [0, 1]]),
                               rtol=1e-10, atol=1e-10)
