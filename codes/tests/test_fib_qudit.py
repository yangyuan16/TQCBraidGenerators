import numpy as np
from codes.braiding_generators.fib_qudit import (
    check_rule,
    check_state,
    find_basis,
    F,
    R,
    B,
    sigma,
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


def test_check_rule():
    """ """
    assert check_rule(1, 1, 1)
    assert check_rule(1, 1, 0)
    assert check_rule(1, 0, 1)
    assert check_rule(0, 0, 0)

    for _ in range(3):
        a_1 = np.random.randint(2)
        a_2 = np.random.randint(2)
        a_3 = np.random.randint(2)
        assert check_rule(a_1, a_2, a_3) == check_rule(a_2, a_1, a_3)


def test_check_state():
    """ """
    assert check_state([1, 1])
    assert check_state([1, 0])
    assert check_state([0, 1])
    assert not check_state([0, 0])

    state_1 = []
    state_0 = []
    state_10 = []
    index = 0
    for _ in range(np.random.randint(3, 7)):
        state_1.append(1)
        state_0.append(0)

        if index % 2:
            state_10.append(1)
        else:
            state_10.append(0)
        index += 1

    assert check_state(state_1)
    assert not check_state(state_0)
    assert check_state(state_10)


def test_find_basis():
    """ """
    # test the length of the basis
    for _ in range(1, 10):
        len(find_basis(_)) == fib_seq(_)

    # test the basis of 3 and 4 anyons
    assert all([_ in [[0, 1], [1, 0], [1, 1]] for _ in find_basis(3)])
    assert all([_ in [[0, 1, 0],
                      [1, 1, 0],
                      [1, 0, 1],
                      [0, 1, 1],
                      [1, 1, 1]] for _ in find_basis(4)])


def test_f():
    """ """
    np.testing.assert_allclose(F(1, 1, 1, 1),
                               np.array([[0.61803399, 0.78615138],
                                         [0.78615138, -0.61803399]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(F(1, 1, 1, 0),
                               np.array([[0, 0], [0, 1]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(F(1, 1, 0, 0),
                               np.array([[0, 1], [0, 0]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(F(1, 0, 0, 0),
                               np.array([[0, 0], [0, 0]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(F(0, 0, 0, 0),
                               np.array([[1, 0], [0, 0]]),
                               rtol=1e-7, atol=1e-7
                               )


def test_r():
    """ """
    np.testing.assert_allclose(R(1, 1),
                               np.array([[-0.80901699-0.58778525j,  0],
                                         [0, -0.30901699+0.95105652j]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(R(1, 0),
                               np.array([[1,  0],
                                         [0, 1]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(R(0, 0),
                               np.array([[1,  0],
                                         [0, 1]]),
                               rtol=1e-7, atol=1e-7
                               )


def test_b():
    """ """
    np.testing.assert_allclose(B(1, 1, 1, 1),
                               np.array([[-0.5 + 3.63271264e-01j,
                                          -0.24293414-7.47674391e-01j],
                                         [-0.24293414-7.47674391e-01j,
                                          -0.61803399+4.32956235e-17j]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(B(1, 1, 1, 0),
                               np.array([[0, 0],
                                         [0, -0.30901699+0.95105652j]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(B(1, 1, 0, 0),
                               np.array([[0, 1],
                                         [0, 0]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(B(1, 1, 0, 0),
                               np.array([[0, 1],
                                         [0, 0]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(B(1, 0, 0, 0),
                               np.array([[0, 0],
                                         [0, 0]]),
                               rtol=1e-7, atol=1e-7
                               )

    np.testing.assert_allclose(B(0, 0, 0, 0),
                               np.array([[1, 0],
                                         [0, 0]]),
                               rtol=1e-7, atol=1e-7
                               )


def test_sigma():
    """ """
    assert np.isclose(sigma(1, [0, 1], [0, 1]),
                      -0.8090169943749473-0.5877852522924732j,
                      rtol=1e-7, atol=1e-7)
    assert np.isclose(sigma(1, [1, 0], [1, 0]),
                      -0.30901699437494734+0.9510565162951536j,
                      rtol=1e-7, atol=1e-7)
    assert np.isclose(sigma(1, [1, 1, 1], [1, 1, 1]),
                      -0.30901699437494734+0.9510565162951536j,
                      rtol=1e-7, atol=1e-7)

    assert np.isclose(sigma(1, [1, 1, 1], [1, 0, 1]),
                      0,
                      rtol=1e-7, atol=1e-7)


def test_braiding_generator():
    """ """
    # verify a numerical value
    sigma_ = np.array(braiding_generator(1, 3, False)[0])
    np.testing.assert_allclose(sigma_,
                               np.array([[-0.30901699+0.95105652j, 0, 0],
                                         [0, -0.80901699-0.58778525j, 0],
                                         [0, 0, -0.30901699+0.95105652j]]),
                               rtol=1e-5, atol=1e-5)

    # verify braiding matrices are unitary
    for index in range(1, 4):
        sigma_ = np.array(braiding_generator(index, 4, False)[0])
        np.testing.assert_allclose(sigma_ @ sigma_.conjugate().T,
                                   np.eye(5),
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
