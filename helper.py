from string import ascii_letters
import numpy as np
import scipy.linalg as sl

DEBUG = False
SEED = None
PROC_OFFSET = 2**12
MAX_THREADS = 8

if DEBUG:
    SEED = 6470

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Get the position of the qubit in the state
# This avoids the iteration through all indexes in:
#  'for i in range(self.n_states) if i & mask != 0'
# Improving the performance from O(2^n) to O(2^(n-1))
# qubit: qubit number
# i: index relative to the list of masked qubits
def pos_0(qubit : int, i : int) -> int:
    mask = 1 << qubit
    return mask*2*(i>>qubit) + (i & (mask - 1))

def pos_1(qubit : int, i : int) -> int:
    return pos_0(qubit, i) + (1 << qubit)

def prob_state_reg(states: np.ndarray, state: int) -> float:
        return (abs(states[state, 0])**2)

def prob_state_dens(states: np.ndarray, state: int) -> float:
    return np.real(states[state, state])

# Create a projector matrix for a given qubit.
def make_projector(n_qubits: int, qubit: int, value: int) -> np.ndarray:
    projector = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

    for i in range(2**(n_qubits-1)):
        index = pos_0(qubit, i) if value == 0 else pos_1(qubit, i)
        projector[index, index] = 1.0 + 0j

    return projector

# Returns the reduced density matrix
# The partial trace is calculated by reshaping the density matrix and using einsum to sum over the traced-out qubits
def partial_trace(matrix : np.ndarray, n_qubits : int, q_index : int) -> np.ndarray:
    dim = 2 * n_qubits

    # Split the density matrix into blocks of 2 x 2
    matrix = np.reshape(matrix, [2] * dim)
    log("matrix.shape\n", matrix.shape)

    # For the matrix reduction, we will use Einstein Summation notation of NumPy
    # In a 8 x 8 matrix (3 qubits)
    # 2i x 2x x 2u x 2j x 2y x 2v
    # ^_______________^            Tr(0) = sum i + j
    #       ^______________^       Tr(1) = sum x + y
    #            ^______________^  Tr(2) = sum u + v
    
    matrix_indices = ascii_letters[0 : dim]
    log("\matrix_indices:", matrix_indices)
    matrix_indices = list(matrix_indices)

    matrix_indices[q_index + n_qubits] = matrix_indices[q_index]
    einsum_indices = "".join(matrix_indices)

    log("einsum_indices:", einsum_indices)
    matrix = np.einsum(einsum_indices, matrix)

    return np.reshape(matrix, (1, 2**(n_qubits - 1), 2**(n_qubits - 1)))[0]

def calculate_entropy(rho : np.ndarray) -> float:
    log_rho = sl.logm(rho, disp=False)[0]
    log("log_rho:\n", log_rho)
    return -np.trace(rho * log_rho).real