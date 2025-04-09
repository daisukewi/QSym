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

def prob_state(states: np.ndarray, state: int) -> float:
    return np.real(states[state, state])

# Create a projector matrix for a given qubit.
def make_projector(n_qubits: int, qubit: int, value: int) -> np.ndarray:
    projector = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

    for i in range(2**(n_qubits-1)):
        index = pos_0(qubit, i) if value == 0 else pos_1(qubit, i)
        projector[index, index] = 1.0 + 0j

    return projector

class QDensity:
    # Initialize the density matrix for a given number of qubits
    # and optional probabilities for the states.
    # prob: list of tuples (probability, state_index)
    def __init__(self, n_qubits: int, prob : List[tuple[float, int]] = None):
        self.n_qubits = n_qubits
        self.density_matrix = np.zeros((self.n_states, self.n_states), dtype=complex)
        self.__rng = np.random.default_rng(SEED)

        if prob is None:
            # initialize the pure state density matrix for the |0><0| state
            self.density_matrix[0, 0] = 1.0+0j
        else:
            # initialize the mixed states density matrix with the given probabilities
            for p in prob:
                self.density_matrix[p[1], p[1]] = p[0]
            
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
    
    def probability(self, qubit: int) -> float:
        sum_prob = 0.0
        for i in range(2**(self.n_qubits-1)):
            log(f"qubit: {qubit}, i: {i}, pos: {pos_1(qubit,i)}")
            sum_prob += prob_state(self.density_matrix, pos_1(qubit,i))
        return max(min(sum_prob, 1.0), 0.0)

    def measure(self, qubits: Union[List[int], int], rand: Callable[[], float] = None) -> list:
        if rand is None:
            rand = self.__rng.random
        if isinstance(qubits, int):
            qubits = [qubits]
        
        values = [0 for _ in range(len(qubits))]
        for qubit in range(len(qubits)):
            prob_one = self.probability(qubits[qubit])
            rnd_value = rand()
            log("Prob:", prob_one, "- Rand:", rnd_value)
            values[qubit] = 1 if rnd_value < prob_one else 0
            
            # Collapse the density matrix to the measured state
            projector = make_projector(self.n_qubits, qubits[qubit], values[qubit])
            trace = np.real(np.trace(self.density_matrix @ projector))
            self.density_matrix = (projector @ self.density_matrix @ projector) / trace

        return values