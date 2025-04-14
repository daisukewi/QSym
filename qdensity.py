import math
import numpy as np
import scipy.sparse as sp

from typing import Callable, Union, List
from gates import Gate
from helper import calculate_entropy, log, SEED, pos_1, make_projector, partial_trace, prob_state_dens

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
            sum_prob += prob_state_dens(self.density_matrix, pos_1(qubit,i))
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
    
    # Calculate the partial trace of the density matrix, removing the specified qubits
    # indices: list of qubit indices to be traced out
    def partial_trace(self, index: int) -> np.ndarray:
        return partial_trace(self.density_matrix, self.n_qubits, index)
    
    def calculate_entropy(self) -> float:
        return calculate_entropy(self.density_matrix)
    
    