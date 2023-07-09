import numpy as np
from braiding_generators.ising_qudit import (
    F,
    R,
    check_rule,
    check_state,
    find_basis,
    iterate,
    B,
    sigma,
    braiding_generator,
)


def test_f():
    """ """
    np.testing.assert_allclose(
        F(1, 1, 1, 1),
        np.array(
            [[0.70710678, 0, 0.70710678],
             [0, 0, 0],
             [0.70710678, 0, -0.70710678]]
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        F(1, 1, 1, 0), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        rtol=1e-6, atol=1e-6
    )


def test_r():
    """ """
    np.testing.assert_allclose(
        R(1, 1),
        np.array(
            [
                [np.exp(-np.pi * 1j / 8), 0, 0],
                [0, 0, 0],
                [0, 0, np.exp(3 * np.pi * 1j / 8)],
            ]
        )
    )


def test_check_rule():
    """ """
    assert check_rule(0, 0, 0)
    assert not check_rule(0, 0, np.random.randint(1, 3))
    assert check_rule(1, 0, 1)
    assert check_rule(2, 0, 2)
    assert check_rule(2, 2, 0)
    assert check_rule(1, 1, 0)
    assert check_rule(1, 1, 2)
    assert check_rule(1, 2, 1)
    assert not check_rule(2, 2, np.random.randint(1, 3))


def test_check_state():
    """ """
    assert check_state([0])
    assert not check_state([1])
    assert check_state([0, 1])
    assert not check_state([0, 0])
    assert check_state([0, 1, 2, 1])
    assert check_state([0, 1, 0, 1])
    assert not check_state([0, 1, 1, 0])


def test_find_basis_check_state():
    """ """
    basis = find_basis(4)
    for base in basis:
        assert check_state(base)

    basis = find_basis(6)
    for base in basis:
        assert check_state(base)


def test_find_basis():
    """ """
    # test the length of the basis
    for _ in range(10):
        assert len(find_basis(_)) == 2**(_//2)

    # test the basis of 3 and 4 anyons
    assert all([_ in [[0, 1], [2, 1]] for _ in find_basis(3)])
    assert all([_ in [[0, 1, 0], [2, 1, 0],
                      [0, 1, 2], [2, 1, 2]] for _ in find_basis(4)])


def test_iterate():
    """ """
    nb_labels = np.random.randint(2, 5)
    iterator = iterate(nb_labels)
    count = 0
    total = np.zeros(nb_labels)
    for itr in iterator:
        count += 1
        total += np.array(itr)

    assert count == 3**nb_labels
    np.testing.assert_approx_equal(np.sum(total) / nb_labels, total[0], 1)


def test_b():
    """ """
    np.testing.assert_allclose(B(1, 1, 1, 1),
                               np.array([[0.65328148+0.27059805j, 0,
                                          0.27059805-0.65328148j],
                                         [0, 0, 0],
                                         [0.27059805-0.65328148j, 0,
                                          0.65328148+0.27059805j]]),
                               rtol=1e-6, atol=1e-6
                               )


def test_sigma():
    """ """
    assert np.isclose(sigma(1, [0, 1, 2, 1], [0, 1, 2, 1]),
                      0.9238795325112867-0.3826834323650898j,
                      rtol=1e-5, atol=1e-5)
    assert np.isclose(sigma(1, [0, 1, 2, 1], [0, 1, 0, 1]), 0)
    assert np.isclose(sigma(2, [0, 1, 2, 1], [2, 1, 2, 1]),
                      0.2705980500730984-0.6532814824381881j,
                      rtol=1e-5, atol=1e-5)
    assert np.isclose(sigma(2, [0, 1, 2], [2, 1, 2]),
                      0.2705980500730984-0.6532814824381881j,
                      rtol=1e-5, atol=1e-5)


def test_braiding_generator():
    """ """
    # verify braiding matrices are unitary
    for index in range(1, 4):
        sigma_ = np.array(braiding_generator(index, 4, False)[0])
        np.testing.assert_allclose(sigma_ @ sigma_.conjugate().T,
                                   np.eye(4),
                                   rtol=1e-5, atol=1e-5)

    # verify a numerical value
    sigma_ = np.array(braiding_generator(1, 4, False)[0])
    np.testing.assert_allclose(sigma_,
                               np.array([[0.92387953-0.38268343j, 0, 0, 0],
                                         [0, 0.38268343+0.92387953j, 0, 0],
                                         [0, 0, 0.92387953-0.38268343j, 0],
                                         [0, 0, 0, 0.38268343+0.92387953j]]),
                               rtol=1e-5, atol=1e-5)

    # verify Artin algebra (Yang-Baxter equations)
    nb_anyons = np.random.randint(4, 7)
    sigmas = []
    for index in range(1, nb_anyons):
        sigmas.append(np.array(braiding_generator(index, nb_anyons, False)[0]))

    for index in range(1, nb_anyons-2):
        np.testing.assert_allclose(
            sigmas[index] @ sigmas[index+1] @ sigmas[index],
            sigmas[index+1] @ sigmas[index] @ sigmas[index+1],
            rtol=1e-5, atol=1e-5
        )
