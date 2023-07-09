"""
Reproduce Bonesteel CNOT gate.
"""
import pickle
import numpy as np

from codes.braid_matrix_calculator import error_distance, get_matrix, leakage_error
from codes.transformer import uncouple, time_mirror, uncouple_all
import codes.braiding_generators.fib_multi_qudits as multi_q
from codes.matrix_tools import extract, combine_diag
from codes.cplot import cplot, scale

tol = 0.01
# precomputed sequence
sequence =  {'sigma': [3, 4, 4, 3, 3, 4, 2, 3, 3, 2, 4, 3, 3, 4, 2, 3, 3, 2, 4,
                       3, 3, 4, 2, 3, 3, 2, 4, 3, 3, 4, 4, 3, 3, 4, 2, 3, 3, 2,
                       4, 3, 3, 4, 4, 3, 3, 4, 2, 3, 3, 2, 2, 3, 3, 2, 4, 3, 3,
                       4, 2, 3, 3, 2, 4, 3, 3, 4, 2, 3, 3, 2, 2, 3, 3, 2, 4, 3, 
                       3, 4, 2, 3, 3, 2, 2, 3, 3, 2, 4, 3, 3, 4, 2, 3, 3, 2, 2,
                       3, 1, 2, 2, 1, 3, 2, 2, 3, 1, 2, 2, 1, 3, 2, 2, 3, 1, 2,
                       2, 1, 3, 2, 2, 3, 3, 2, 2, 3, 1, 2, 2, 1, 3, 2, 2, 3, 3,
                       2, 2, 3, 1, 2, 2, 1, 3, 2, 2, 3, 3, 2, 2, 3, 1, 2, 2, 1,
                       3, 2, 2, 3, 1, 2, 2, 1, 3, 2, 2, 3, 1, 2, 2, 1, 1, 2, 2,
                       1, 3, 2, 2, 3, 3, 2, 2, 3, 1, 2, 2, 1, 3, 2, 2, 3, 3, 2, 
                       4, 3, 3, 4, 2, 3, 3, 2, 2, 3, 3, 2, 4, 3, 3, 4, 2, 3, 3,
                       2, 2, 3, 3, 2, 4, 3, 3, 4, 2, 3, 3, 2, 4, 3, 3, 4, 2, 3,
                       3, 2, 2, 3, 3, 2, 4, 3, 3, 4, 4, 3, 3, 4, 2, 3, 3, 2, 4,
                       3, 3, 4, 4, 3, 3, 4, 2, 3, 3, 2, 4, 3, 3, 4, 2, 3, 3, 2,
                       4, 3, 3, 4, 2, 3, 3, 2, 4, 3, 3, 4, 4, 3], 
             'power': [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1,
                       1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 
                       1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 
                       -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1,
                       1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1,
                       1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1,
                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1,
                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
                       -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                       1, 1, 1, 1, -1, -1, -1, -1, -1, -1]}

# Calculate sigmas of 2 qubits (3 anyons per qubit)
try:
    with open("bin/basis-2q-6a.pickle", "rb") as file:
        basis = pickle.load(file)

    with open("bin/SIG-2q-6a.pickle", "rb") as file:
        SIG = pickle.load(file)

except FileNotFoundError:
    SIG = {}
    for index in range(5):
        n = index + 1
        SIG[n] = {}
        gen = multi_q.braiding_generator(n, 2, 2, show=False)

        SIG[n][1] = np.array(gen[0])
        SIG[n][-1] = np.array(np.linalg.inv(gen[0]))

    basis = gen[1]
    with open("bin/basis-2q-6a.pickle", "wb") as file:
        pickle.dump(basis, file)
    with open("bin/SIG-2q-6a.pickle", "wb") as file:
        pickle.dump(SIG, file)

#
# Calculate matrix representation of C-NOT braiding sequance
w = get_matrix(sequence, sigma=SIG)#.round(7)

# Extract computational submatrix (either of total charge 0 or 1)
w_0 = extract(w, [1, 2, 3, 4])
w_1 = extract(w, [9, 10, 11, 12])

# Calculate exact C-NOT matrix
NOT = np.array([[0, 1], [1, 0]]) * (0 + 1j)
C_NOT = np.kron([[1, 0], [0, 0]], np.eye(2)) + np.kron([[0, 0], [0, 1]], NOT)

C_NOT_13 = combine_diag(combine_diag(C_NOT, np.eye(1)),
                        combine_diag(C_NOT, np.eye(4)))

# Show results

print('basis:\n')
print(basis)
#print('\n Exact 13x13 C-NOT matrix representation =')
#cplot(array(C_NOT))
#cplot(np.array(C_NOT_13), sigma=tol)

print('\n 13x13 braiding C-NOT matrix representation =')
cplot(np.array(w), title=f"bonesteel-cnot-full", sigma=tol, show=False)
print('Notice the correlation between computational states\n'\
      'and non-computational states.')

# Calculate only 2x2 NOT submatrix from 4x4 matrix
#x_0 = extract(w, [3, 4])
#x_1 = extract(w, [11, 12])

print('\n Exact C-NOT matrix representation =')
cplot(np.array(C_NOT), title=f"cnot-exact", sigma=tol, show=False)


# show C_NOT braiding matrix in total charge 0
print('\n C-NOT braid gate matrix (with total charge 0) = \n')
cplot(np.array(w_0), title=f"bonesteel-cnot-0", sigma=tol, show=False, ticks=[1, 2, 3, 4])
print(f'leakage error = {leakage_error(np.array(w_0))}')
print('Error = ', error_distance(w_0, C_NOT))

# show C_NOT braiding matrix in total charge 1
print('\n C-NOT braid gate matrix (with total charge 1) = \n')
cplot(np.array(w_1), title=f"bonesteel-cnot-1", sigma=tol, show=False, ticks=[9, 10, 11, 12])
print(f'leakage error = {leakage_error(np.array(w_1))}')
print('Error = ', error_distance(w_1, C_NOT))

scale(title='scale', sigma=tol, show=False)

# Show sequence of braiding sigmas
print('\n Braid sequence = ', sequence)

# Show braiding generators
for n in range(1, 6):
    cplot(SIG[n][1],
          title=f"sigma2q3a_{n}",
          sigma=tol, show=False)
    cplot(SIG[n][-1],
          title=f"sigma2q3a_{n}_inv",
          sigma=tol, show=False)
