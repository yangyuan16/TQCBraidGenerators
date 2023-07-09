from codes.transformer import (
    describe,
    uncouple,
    uncouple_all,
    time_mirror,
    seq_to_latex
)


def test_describe():
    """ """
    seq = describe([2, 2], 1, 1, 1)
    assert {'sigma': [1, 1, 1, 2, 2, 1],
            'power': [1, 1, 1, 1, 1, 1],
            'strand_rank': [1, 2, 1, 2, 3, 2, 1]}

    seq = describe([2, 2], 1, 1, -1)
    assert {'sigma': [1, 1, 1, 2, 2, 1],
            'power': [1, 1, 1, 1, 1, -1],
            'strand_rank': [1, 2, 1, 2, 3, 2, 1]}

    seq = describe([2, 2], 1, 1, 3)
    assert seq == {'sigma': [1, 1, 1, 2, 2, 2],
                   'power': [1, 1, 1, 1, 1, 1],
                   'strand_rank': [1, 2, 1, 2, 3, 2, 3]}

    seq = describe([2, 2], 1, -3, 3)
    assert seq == {'sigma': [2, 1, 1, 2, 2, 2],
                   'power': [-1, 1, 1, 1, 1, 1],
                   'strand_rank': [3, 2, 1, 2, 3, 2, 3]}

    assert describe([2, 2], 1, 1, 2) == describe([2, 2], 1, 1, -2)


def test_uncouple():
    """ """
    seq = uncouple([2, 2], 1, 1, 1, rank_increment=0, n_coupled_strands=2)
    assert seq == {'sigma': [2, 1, 1, 2, 2, 1, 3, 2, 2, 3, 1, 2],
                   'power': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    seq = uncouple([2, 2], 1, 1, 3, rank_increment=0, n_coupled_strands=2)
    assert seq == {'sigma': [2, 1, 1, 2, 2, 1, 3, 2, 2, 3, 3, 2],
                   'power': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    seq = uncouple([2, 2], 1, -3, 3, rank_increment=0, n_coupled_strands=2)
    assert seq == {'sigma': [2, 3, 1, 2, 2, 1, 3, 2, 2, 3, 3, 2],
                   'power': [-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    seq = uncouple([2, 2], 1, 1, 2, rank_increment=0, n_coupled_strands=2)
    assert seq == {'sigma': [2, 1, 1, 2, 2, 1, 3, 2, 2, 3],
                   'power': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    seq = uncouple([2, 2], 1, 1, 2, rank_increment=1, n_coupled_strands=2)
    assert seq == {'sigma': [3, 2, 2, 3, 3, 2, 4, 3, 3, 4],
                   'power': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    seq = uncouple([2, 2], 1, 1, 2, rank_increment=0, n_coupled_strands=3)
    assert seq == {'sigma': [3, 2, 1, 1, 2, 3, 3, 2, 1, 4, 3, 2, 2, 3, 4],
                   'power': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


def test_uncouple_all():
    """ """
    seq = uncouple_all([2, 2], 1, 1, 2, rank_increment=0)
    assert seq == {'sigma': [2, 1, 3, 2, 2, 1, 3, 2, 2,
                             1, 3, 2, 4, 3, 5, 4, 4, 3, 5, 4],
                   'power': [1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    seq = uncouple_all([2, 2], 1, 1, 1, rank_increment=0)
    assert seq == {'sigma': [2, 1, 3, 2, 2, 1, 3, 2, 2,
                             1, 3, 2, 4, 3, 5, 4, 4, 3, 5, 4, 2, 1, 3, 2],
                   'power': [1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


def test_time_mirror():
    """ """
    seq = {'sigma': [1, 2, 1], 'power': [2, 2, 2]}
    assert time_mirror(seq) == {'sigma': [1, 2, 1], 'power': [-2, -2, -2]}

    seq = {'sigma': [1, 2, 1, 2], 'power': [2, 2, 2, 2]}
    assert time_mirror(seq) == {'sigma': [2, 1, 2, 1],
                                'power': [-2, -2, -2, -2]}


def test_seq_to_latex():
    """ """
    expression = seq_to_latex([2, 2], 1, 1, 1)
    assert expression == '\\sigma_1^{1}\\sigma_2^{2}\\sigma_1^{2}\\sigma_1^{1}'

    expression = seq_to_latex([2, 2], 1, 1, 3)
    assert expression == '\\sigma_2^{1}\\sigma_2^{2}\\sigma_1^{2}\\sigma_1^{1}'

    expression = seq_to_latex([2, 2], 2, 3, 3)
    assert expression == '\\sigma_2^{1}\\sigma_1^{2}\\sigma_2^{2}\\sigma_2^{1}'
