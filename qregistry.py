import math
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
from typing import Callable, Union, List
from gates import Gate
from helper import DEBUG, PROC_OFFSET, log, SEED, MAX_THREADS, pos_1, prob_state_reg

class QRegistry:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state = np.zeros((self.n_states, 1), dtype=complex)
        self.state[0] = 1.0+0j
        self.__rng = np.random.default_rng(SEED)
        self.pool = Pool(MAX_THREADS)
        log(f"QRegistry Size: {self.n_states}")

    @property
    def n_states(self) -> int:
        return 2**self.n_qubits
    
    def get_state(self) -> np.array:
        return self.state
    
    def density_matrix(self) -> np.array:
        return np.outer(self.state, np.conj(self.state))
    
    # Time: Θ(2^2n)
    # Space: Θ(2^2n)
    def __apply_gate(self, matrix: sp.coo_matrix, target: int):
        log("Gate Size:", matrix.shape[0])
        qbefore = int(target - (math.log2(matrix.shape[0]) - 1))
        log("before:", qbefore)
        gate_matrix = sp.kron(matrix, sp.identity(2**qbefore, dtype=complex), format="coo")

        if target < self.n_qubits - 1:
            gate_matrix = sp.kron(sp.identity(2**(self.n_qubits - target - 1), dtype=complex), gate_matrix, format="coo")

        log("dim:", gate_matrix.shape, "-", self.state.shape)
        
        #scipy multiply using dot() does a pretty bad job compared to the @ operator
        #self.state = gate_matrix.dot(self.state)
        self.state = gate_matrix @ self.state

        if DEBUG:
            log(self.get_state())
            for i in range(self.n_qubits):
                log(self.probability(i), end=" ")
            log("\n")
    
    def apply_gates(self, gates: Union[List[Gate], Gate], targets: Union[List[int], int]):
        if isinstance(targets, int):
            targets = [targets]
        if isinstance(gates, Gate):
            gates = [gates] * len(targets)

        if len(targets) == 1:
            return self.__apply_gate(gates[0].matrix, targets[0])

        gate_matrix = sp.coo_matrix([1])

        # Sort the gates and targets by target index
        # This is important to avoid the creation of a large sparse matrix
        gates_list = sorted(list(zip(gates, targets)), key=lambda x: x[1])

        for i in range(len(gates_list)):
            gate, target = gates_list[i]
            gates_between = int(target - gates_list[i-1][1] - (math.log2(gate.matrix.shape[0]) - 1) - 1) if i > 0 else 0
            log("between:", gates_between)
            gate_matrix = sp.kron(sp.kron(gate.matrix, Gate.I(gates_between).matrix, format=gate_matrix.format), gate_matrix, format=gate_matrix.format)
        
        self.__apply_gate(gate_matrix, gates_list[-1][1])
    
    # Time: O(2^(n-1))
    def probability(self, qubit: int) -> float:
        sum_prob = 0.0
        for i in range(2**(self.n_qubits-1)):
            log(f"qubit: {qubit}, i: {i}, pos: {pos_1(qubit,i)}")
            sum_prob += prob_state_reg(self.state, pos_1(qubit,i))
        return max(min(sum_prob, 1.0), 0.0)
    
    @staticmethod
    def parallel_prob_calc(qubit: int, offset: int, max_size : int, state: np.ndarray) -> float:
        sum_prob = 0.0
        qubit_range = PROC_OFFSET if offset + PROC_OFFSET < max_size else max_size - offset

        for i in range(qubit_range):
            log(f"qubit: {qubit}, i: {offset + i}, pos: {pos_1(qubit, offset + i)}")
            sum_prob += prob_state_reg(state, pos_1(qubit, offset + i))
        
        return sum_prob
    
    def parallel_probability(self, qubit: int) -> float:
        max_size = 2**(self.n_qubits-1)
        n_iter = max_size//PROC_OFFSET + (1 if max_size%PROC_OFFSET != 0 else 0)

        sum_prob = sum(self.pool.starmap(self.parallel_prob_calc, ((qubit, i, max_size, self.state) for i in range(0, n_iter * PROC_OFFSET, PROC_OFFSET))))
        
        return max(min(sum_prob, 1.0), 0.0)
    
    # Time: O(2^n)
    def __collapse(self, qubit: int, value: int, prob_one: float = 0.0):
        value = 1 if value != 0 else 0
        for i in range(self.n_states):
            self.state[i] = self.state[i] * (1.0 if (i >> qubit & 1) == value else 0.0) / math.sqrt(prob_one if value > 0 else 1.0 - prob_one)
    
    @staticmethod
    def parallel_collapse_calc(qubit: int, offset: int, value: int, prob_one: float, max_size : int, state: np.ndarray):
        qubit_range = PROC_OFFSET if offset + PROC_OFFSET < max_size else max_size - offset

        for i in range(qubit_range):
            state[i+offset] = state[i+offset] * (1.0 if (i >> qubit & 1) == value else 0.0) / math.sqrt(prob_one if value > 0 else 1.0 - prob_one)
    
    def __parallel_collapse(self, qubit: int, value: int, prob_one: float = 0.0):
        value = 1 if value != 0 else 0
        max_size = self.n_states
        n_iter = max_size//PROC_OFFSET + (1 if max_size%PROC_OFFSET != 0 else 0)

        self.pool.starmap(self.parallel_collapse_calc, ((qubit, i, value, prob_one, max_size, self.state) for i in range(0, n_iter * PROC_OFFSET, PROC_OFFSET)))

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
            self.__collapse(qubits[qubit], values[qubit], prob_one)

        return values
    
    def parallel_measure(self, qubits: Union[List[int], int], rand: Callable[[], float] = None) -> list:
        if rand is None:
            rand = self.__rng.random
        if isinstance(qubits, int):
            qubits = [qubits]
        
        values = [0 for _ in range(len(qubits))]
        for qubit in range(len(qubits)):
            prob_one = self.parallel_probability(qubits[qubit])
            rnd_value = rand()
            log("Prob:", prob_one, "- Rand:", rnd_value)
            values[qubit] = 1 if rnd_value < prob_one else 0
            self.__parallel_collapse(qubits[qubit], values[qubit], prob_one)

        return values
    
    # Calculate the Bloch Sphere angles
    # from the State Vector |ψ⟩ = α|0⟩ + β|1⟩
    # can also be represented as |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    # taking out the global phase multiplying by α* / |α|
    # return: θ -> Polar angle, φ -> Azimuth angle
    def bloch_angles(self) -> tuple[float, float]:
        if self.n_qubits != 1:
            raise ValueError("Bloch sphere can only be calculated for a single qubit")
        
        alpha = self.state[0, 0]
        beta = self.state[1, 0]

        log(f"alpha: {alpha}, beta: {beta} \n")

        theta = 2 * np.arccos(np.abs(alpha))
        phi = np.angle(beta * np.conjugate(alpha) / (np.abs(alpha) * np.abs(beta)))

        if np.isnan(phi):
            phi = 0
        if phi < 0:
            phi += 2 * np.pi

        log(f"Bloch Sphere |-> φ = {phi:0.4f} - θ = {theta:0.4f}\n")
        return theta, phi
