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
