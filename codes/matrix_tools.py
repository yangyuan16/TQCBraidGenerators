import numpy as np


def extract(matrix_: 'np.matrix', ranks):
    """
    extracts from matrix, nxn submatrix corresponding to indices specified
    in ranks.
    Inputs:
        matrix: numpy.matrix
        ranks: list
    Returns:
        numpy.matrix
    """
    new_matrix = []
    for ii, row in enumerate(ranks):
        new_matrix.append([])
        for jj, column in enumerate(ranks):
            new_matrix[ii].append(matrix_[row, column])
    
    return np.matrix(new_matrix)
    
def combine_diag(m0: 'np.matrix', m1: 'np.matrix'):
    """
    combines two sub-matrices diagonally to form larger matrix.
    Example:
        m0 = np.matrix([[1, 0],
                        [0, 1]])
        m1 = np.matrix([[2, 0],
                        [0, 2]])
        combine_diag(m0, m1)= np.matrix([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 2, 0],
                                         [0, 0, 0, 2]])
        
    """
    # Getting the size
    dim0 = np.shape(m0)[0]
    dim1 = np.shape(m1)[0]
    
    dim_tot = dim0 + dim1

    
    # Allocate zeros to result
    result = []
    for ii in range(dim_tot):
        result.append([])
        for jj in range(dim_tot):
            result[ii].append(0 + 0j)

    
    # Replace values
    for ii in range(dim0):
        for jj in range(dim0):
            result[ii][jj] = m0[ii, jj]
    
    for ii in range(dim1):
        for jj in range(dim1):
            result[dim0 + ii][dim0 + jj] = m1[ii, jj]

    
    return np.matrix(result)

a = np.matrix([[1, 0], [0, 1]])

b = np.matrix([[2, 0], [0, 2]])