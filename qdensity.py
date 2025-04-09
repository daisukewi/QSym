import math
import numpy as np
import scipy.sparse as sp
from typing import Callable, Union, List
from gates import Gate

DEBUG = False
SEED = None
PROC_OFFSET = 2**12
MAX_THREADS = 8

if DEBUG:
    SEED = 6470

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class QDensity:
    def __init__(self, n_qubits: int, prob : List[float] = None):
        self.n_qubits = n_qubits
        self.density_matrix = np.zeros((self.n_states, self.n_states), dtype=complex)
        self.density_matrix[0, 0] = 1.0+0j
        self.__rng = np.random.default_rng(SEED)

        # for p in prob:
        #     rnd_value = self.__rng.random()
        #     self.density_matrix[p[1], p[1]] = 1 if rnd_value < p[0] else 0
            

    @property
    def n_states(self) -> int:
        return 2**self.n_qubits
    
    @property
    def get_density(self) -> np.ndarray:
        return self.density_matrix
    
    def __apply_gate(self, matrix: sp.coo_matrix, target: int):
        log("Gate Size:", matrix.shape[0])
        qbefore = int(target - (math.log2(matrix.shape[0]) - 1))
        log("before:", qbefore)
        gate_matrix = sp.kron(matrix, sp.identity(2**qbefore, dtype=complex), format="coo")

        if target < self.n_qubits - 1:
            gate_matrix = sp.kron(sp.identity(2**(self.n_qubits - target - 1), dtype=complex), gate_matrix, format=gate_matrix.format)

        log("dim:", gate_matrix.shape, "-", self.get_density.shape)
        log("Gate Matrix:\n", gate_matrix.todense())

        self.density_matrix = gate_matrix @ self.density_matrix @ gate_matrix.getH()

        log("Density Matrix:\n", self.density_matrix)
    
    def apply_gates(self, gates: Union[List[Gate], Gate], targets: Union[List[int], int]):
        if isinstance(targets, int):
            targets = [targets]
        if isinstance(gates, Gate):
            gates = [gates] * len(targets)

        if len(targets) == 1:
            return self.__apply_gate(gates[0].matrix, targets[0])

        gate_matrix = sp.coo_matrix([1])
        gates_list = sorted(list(zip(gates, targets)), key=lambda x: x[1])
        for i in range(len(gates_list)):
            gate, target = gates_list[i]
            gates_between = int(target - gates_list[i-1][1] - (math.log2(gate.matrix.shape[0]) - 1) - 1) if i > 0 else 0
            log("between:", gates_between)
            gate_matrix = sp.kron(sp.kron(gate.matrix, Gate.I(gates_between).matrix, format=gate_matrix.format), gate_matrix, format=gate_matrix.format)
        
        self.__apply_gate(gate_matrix, gates_list[-1][1])