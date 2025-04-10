from string import ascii_letters
from typing import List
import numpy as np

def make_mtx(values : np.ndarray) -> np.ndarray:
    return np.array(values, dtype=complex)

class Gate():
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
    
    @matrix.setter
    def matrix(self, matrix: np.ndarray):
        self._matrix = matrix

    @classmethod
    def I(cls, n: int) -> 'Gate':
        return cls(np.eye(2**n))
    
    @classmethod
    def H(cls) -> 'Gate':
        return cls(make_mtx(np.array([[1, 1], [1, -1]]) * np.sqrt(2) / 2))
    
    @classmethod
    def X(cls) -> 'Gate':
        return cls(make_mtx([[0, 1], [1, 0]]))
    
    @classmethod
    def Y(cls) -> 'Gate':
        return cls(make_mtx([[0, -1j], [1j, 0]]))
    
    @classmethod
    def Z(cls) -> 'Gate':
        return cls(make_mtx([[1, 0], [0, -1]]))
    
    @classmethod
    def S(cls) -> 'Gate':
        return cls(make_mtx([[1, 0], [0, 1j]]))
    
    @classmethod
    def T(cls) -> 'Gate':
        return cls(make_mtx([[1, 0], [0, np.exp(1j*np.pi/4)]]))

    @classmethod
    def CNOT(cls) -> 'Gate':
        return cls(make_mtx([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]))
    
    @classmethod
    def SWAP(cls) -> 'Gate':
        return cls(make_mtx([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))

    @classmethod
    def RX(cls, theta: float) -> 'Gate':
        return cls(make_mtx([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]]))
    
    @classmethod
    def RY(cls, theta: float) -> 'Gate':
        return cls(make_mtx([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]]))
    
    @classmethod
    def RZ(cls, theta: float) -> 'Gate':
        return cls(make_mtx([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]]))
    
    @classmethod
    def U(cls, theta: float, phi: float, lambda_: float) -> 'Gate':
        return cls(make_mtx([[np.cos(theta/2), -np.exp(1j*lambda_)*np.sin(theta/2)], [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lambda_))*np.cos(theta/2)]]))

# Returns the reduced density matrix
# The partial trace is calculated by reshaping the density matrix and using einsum to sum over the traced-out qubits
def partial_trace(matrix : np.ndarray, num_indices : int, indices : List[int]) -> np.ndarray:
    rho_dim = 2 * num_indices

    # Split the density matrix into blocks of 2 x 2
    matrix = np.reshape(matrix, [1] + [2] * rho_dim)
    print("matrix\n", matrix)
    print("matrix.shape:", matrix.shape)
    print("split:", [1] + [2] * rho_dim)

    for i, q_index in enumerate(indices):
        q_index = q_index - i
        state_indices = ascii_letters[1 : rho_dim - 2 * i + 1]
        print("\nstate_indices:", state_indices, "q_index:", q_index, "i:", i)
        state_indices = list(state_indices)

        target_letter = state_indices[q_index]
        state_indices[q_index + num_indices - i] = target_letter
        state_indices = "".join(state_indices)
        print("target_letter:", target_letter)
        print("state_indices:", state_indices)

        einsum_indices = f"a{state_indices}"
        print("einsum_indices:", einsum_indices)
        matrix = np.einsum(einsum_indices, matrix)
        print("matrix\n", matrix)

    number_wires_sub = num_indices - len(indices)
    print("number_wires_sub:", number_wires_sub)
    reduced_density_matrix = np.reshape(matrix, (1, 2**number_wires_sub, 2**number_wires_sub))
    return reduced_density_matrix[0]