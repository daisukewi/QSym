# Quantum Circuit Simulator - QRegistry Usage

This document describes how to use the `QRegistry` class, a Python class for simulating quantum states and circuits based on state vectors.

This project is a quantum simulator that implements both state vector and density matrix representations for quantum systems. It allows users to define and manipulate quantum states and apply quantum gates to simulate the evolution of quantum systems.

## Overview

The `QRegistry` class manages the state vector of a quantum system composed of a specified number of qubits. It allows you to:

*   Initialize a quantum state (defaults to the |0...0⟩ state).
*   Apply quantum gates to specific qubits.
*   Apply multiple gates simultaneously across different qubits.
*   Calculate the probability of measuring a specific qubit in the |1⟩ state.
*   Simulate the measurement of one or more qubits, collapsing the state vector accordingly.
*   Retrieve the current state vector.
*   Calculate the density matrix for the pure state.
*   Calculate Bloch sphere angles for single-qubit systems.
*   Utilize multiprocessing for potentially faster probability calculations and measurements on systems with many qubits.

## Prerequisites

*   **Python 3**
*   **NumPy:** For numerical operations, especially vector/matrix manipulation.
    ```bash
    pip install numpy
    ```
*   **SciPy:** Used for sparse matrix operations when applying gates.
    ```bash
    pip install scipy
    ```
*   **Matplotlib:** A comprehensive library for creating static, animated, and interactive visualizations in Python.
    ```bash
    pip install matplotlib
    ```
*   **`gates.py`:** This file must be present in the same directory or accessible in your Python path. It defines the `Gate` class and standard quantum gates (H, X, CNOT, etc.).

## Getting Started

### 1. Import Necessary Classes

```python
import numpy as np
from qregistry import QRegistry
from gates import Gate
```

### 2. Initialize the Registry

Create an instance of QRegistry by specifying the number of qubits. The state is automatically initialized to |0...0⟩.

```python
# Create a 3-qubit quantum registry
num_qubits = 3
registry = QRegistry(num_qubits)

# Check the initial state (optional)
# Output will be a column vector with 1.0 at the top and 0.0 elsewhere
print("Initial State:\n", registry.get_state())
```

### 3. Apply Gates

Use the apply_gates method. You need Gate objects from the gates.py module.

* **Single Gate Application:**

```python
# Apply Hadamard gate to qubit 0
registry.apply_gates(Gate.H(), 0)

# Apply Pauli-X gate to qubit 1
registry.apply_gates(Gate.X(), 1)

print("State after H(0) and X(1):\n", registry.get_state())
```

* **Multi-Qubit Gate Application (e.g., CNOT):** CNOT and other multi-qubit gates defined in `gates.py` (like `SWAP`) are applied by specifying the last target qubit index. The gate definition itself implies which qubits it acts upon relative to this target. For example, `Gate.CNOT()` produces a 4x4 matrix (acting on 2 qubits). Applying it to target `1` means it acts on qubits `0` (control) and `1` (target). Applying it to target `2` means it acts on qubits `1` (control) and `2` (target).

```python
# Apply CNOT gate with qubit 0 as control and qubit 1 as target
# The target parameter refers to the *last* qubit the gate acts on.
registry.apply_gates(Gate.CNOT(), 1)

print("State after CNOT(0, 1):\n", registry.get_state())
```

* **Applying Multiple Gates Simultaneously:** You can apply different gates to different qubits in a single step by providing lists of gates and targets. The gates are applied logically as a tensor product.

```python
# Apply H to qubit 0 and X to qubit 2 simultaneously
registry.apply_gates([Gate.H(), Gate.X()], [0, 2])

print("State after H(0) and X(2):\n", registry.get_state())
```

* **Applying The Same Gate to Multiple Qubits:** A single gate can be applied to multiple qubits at once to save applying multiple tensor products. The list of targets are the target qubits to apply the gate to.

```python
# Apply H to qubits 0, 1, and 2 simultaneously
registry.apply_gates(Gate.H(), [0, 1, 2])

print("State after H(0), H(1), H(2):\n", registry.get_state())
```

### 4. Measure Qubits

Simulate measurement using the `measure` or `parallel_measure` method. This method returns the measurement outcome (0 or 1 for each measured qubit) and collapses the state vector according to the outcome.

```python
# Initialize a 2-qubit registry
registry = QRegistry(2)

# Create a Bell state |Φ+⟩ = (|00⟩ + |11⟩) / sqrt(2)
registry.apply_gates(Gate.H(), 0)
registry.apply_gates(Gate.CNOT(), 1)
print("State before measurement:\n", registry.get_state())

# Measure both qubits (0 and 1)
# A random number generator can be optionally provided
measurement_results = registry.measure([0, 1])
print("Measurement results:", measurement_results)

# The state vector is now collapsed to the measured state (e.g., |00⟩ or |11⟩)
print("State after measurement:\n", registry.get_state())
```

### 5. Calculate Probabilities

Calculate the probability of measuring a specific qubit in the |1⟩ state without collapsing the state.

```python
# Initialize a 1-qubit registry
registry = QRegistry(1)

# Put it in state |+⟩ = (|0⟩ + |1⟩) / sqrt(2)
registry.apply_gates(Gate.H(), 0)

# Calculate the probability of measuring qubit 0 as '1'
prob_1 = registry.probability(0) # or registry.parallel_probability(0)
print(f"Probability of measuring |1⟩ on qubit 0: {prob_1:.4f}") # Should be ~0.5
```

## Key Methods Reference

* `__init__(self, n_qubits: int)`: Constructor. Initializes a registry with n_qubits in the |0...0⟩ state. Time complexity is O(2<sup>n</sup>).
* `get_state(self) -> np.array`: Returns the current state vector as a NumPy complex array of shape (2<sup>n_qubits</sup>). Time complexity is O(1).
* `apply_gates(self, gates: Union[List[Gate], Gate], targets: Union[List[int], int])`: Applies one or more gates to the specified target qubits. Modifies the internal state vector. Internally uses sparse matrix operations for multi-qubit gates. This saves memory and we can reduce time complexity from Θ(n<sup>3</sup>) to Θ(2<sup>n</sup>).
* `apply_c_gate(self, gate: Gate, target: int, control: int)`: Applies a controlled gate operation on the selected target qubit register. Handles the necessary SWAP operations to align the qubits for the gate application. Time complexity is O(n*2<sup>n</sup>)
* `probability(self, qubit: int) -> float`: Calculates the probability P(1) for the given qubit index without modifying the state. Instead of going through all the states, it only access the states where the qubit is in the |1⟩ state, reducing the time complexity from Θ(2<sup>n</sup>) to Θ(2<sup>n-1</sup>).
* `parallel_probability(self, qubit: int) -> float`: A parallelized version of probability, potentially faster for large numbers of qubits. Uses the multiprocessing pool initialized in the constructor. `Note: It takes longer than using the single thread version 😂.`
* `measure(self, qubits: Union[List[int], int], rand: Callable[[], float] = None) -> list`: Simulates measuring the specified qubit(s). Returns a list of outcomes (0 or 1). Collapses the state vector. Optionally accepts a custom random number generating function rand (which should return floats between 0.0 and 1.0). Time complexity is O(2<sup>n</sup> * m) (where m is the number of qubits to measure).
* `parallel_measure(self, qubits: Union[List[int], int], rand: Callable[[], float] = None) -> list`: A parallelized version of measure. Uses parallel_probability and a parallelized collapse mechanism. Collapses the state vector.  `Note: Surprisingly, this actually takes less time than the single thread version 😲.`
* `density_matrix(self) -> np.array`: Calculates and returns the density matrix ρ = |ψ⟩⟨ψ| for the current state vector |ψ⟩. Returns a NumPy array of shape (2<sup>n_qubits</sup>, 2<sup>n_qubits</sup>).
* `print_probabilities(self, until : int = None)`: Prints the probabilities of each state. Useful for debugging. Time complexity is O(2<sup>n</sup>).
* `bloch_angles(self) -> tuple[float, float]`: For single-qubit systems only (n_qubits=1), calculates the polar angle θ and azimuthal angle φ representing the state on the Bloch sphere. Returns (theta, phi). Time complexity is O(1).
* `draw_bloch_sphere(self, filename: str = None)`: Draws the Bloch sphere representation of the single qubit state using matplotlib. If a filename is provided, it saves the figure image to a file. Time complexity is O(1).

## Example: Creating and Measuring a Bell State

```python
import numpy as np
from qregistry import QRegistry
from gates import Gate

# 1. Initialize a 2-qubit registry
registry = QRegistry(2)
print("Initial State:\n", registry.get_state())

# 2. Apply Hadamard gate to qubit 0
registry.apply_gates(Gate.H(), 0)
print("\nState after H(0):\n", registry.get_state())

# 3. Apply CNOT gate (control=0, target=1)
# Remember: target index is the *last* qubit the gate acts on
registry.apply_gates(Gate.CNOT(), 1)
print("\nState after CNOT(0, 1) (Bell state |Φ+⟩):\n", registry.get_state())

# 4. Check probabilities (optional)
prob0 = registry.probability(0)
prob1 = registry.probability(1)
print(f"\nP(1) for qubit 0: {prob0:.2f}") # Expect 0.5
print(f"P(1) for qubit 1: {prob1:.2f}") # Expect 0.5

# 5. Measure both qubits
print("\nMeasuring qubits [0, 1]...")
# Use the default random number generator
results = registry.measure([0, 1])
print("Measurement outcome:", results) # Expect [0, 0] or [1, 1]

# 6. Check the state after measurement
print("\nState after measurement:\n", registry.get_state()) # Expect |00⟩ or |11⟩
```

## More Examples

The file `main.py` contains some test batteries that I have used during the develpment of the project. The code is easy to read, so feel free to use it to get some inspiration on how to use the `QRegistry` class.

## Notes on Parallelism

* The `parallel_probability` and `parallel_measure` methods attempt to speed up calculations using Python's `multiprocessing.Pool`.
* The number of worker processes can be adjusted by changing `MAX_THREADS` within `qregistry.py`.
* The overhead of creating processes might make the parallel versions slower for systems with a small number of qubits. Performance benefits are typically seen for larger `n_qubits`.
* The `PROC_OFFSET` constant in `qregistry.py` controls the chunk size for parallel processing tasks.
* `MAX_THREADS` & `PROC_OFFSET` must be tweaked for better results... but I don't have enough time to try every combination.

## License

This project is licensed under the MIT License. Because every project needs to have a licence 😉.