# Quantum Circuit Simulator

This project is a quantum simulator that implements both state vector and density matrix representations for quantum systems. It allows users to define and manipulate quantum states and apply quantum gates to simulate the evolution of quantum systems.

This document describes how to use the `QRegistry` class, a Python class for simulating quantum states and circuits based on state vectors. Alternativelly you can use the `QDensity` class, that provides quantum simulation using a density matrix approach.

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
*   **Matplotlib:** Used to plot graphics to helps visualize quantum states in the Bloch Sphere.
    ```bash
    pip install matplotlib
    ```
*   **`gates.py`:** This file must be present in the same directory or accessible in your Python path. It defines the `Gate` class and standard quantum gates (H, X, CNOT, etc.).

# QRegistry Usage

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

* **Applying a controlled gate to a qubit:** A single controlled gate can be applied to a single qubit, specifiying the control and target qubits. The method automatically applies the necessary SWAP operations to align the qubits for the gate application, taking into account the gate size.

```python
# Apply H to qubits 0 and a controlled gate to qubit 4 with control 0
registry = QRegistry(5)
registry.apply_c_gate(Gate.H(), 0)
registry.apply_c_gate(Gate.CNOT(), 4, 0)

print("State after H(0), CNOT(0, 4):\n", registry.get_state())
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

* `__init__(self, n_qubits: int)`: Constructor. Initializes a registry with n_qubits in the |0...0⟩ state. Time complexity is $\mathcal{O}(2^{n})$.
* `get_state(self) -> np.array`: Returns the current state vector as a NumPy complex array of shape (2<sup>n_qubits</sup>). Time complexity is $\mathcal{O}(1)$.
* `apply_gates(self, gates: Union[List[Gate], Gate], targets: Union[List[int], int])`: Applies one or more gates to the specified target qubits. Modifies the internal state vector. Internally uses sparse matrix operations for multi-qubit gates. This saves memory and we can reduce time complexity from $\mathcal{O}(3^{n})$ to $\mathcal{O}(2^{n})$.
* `apply_c_gate(self, gate: Gate, target: int, control: int)`: Applies a controlled gate operation on the selected target qubit register. Handles the necessary SWAP operations to align the qubits for the gate application. Time complexity is $\mathcal{O}(n·2^{n})$
* `probability(self, qubit: int) -> float`: Calculates the probability P(1) for the given qubit index without modifying the state. Instead of going through all the states, it only access the states where the qubit is in the |1⟩ state, reducing the time complexity from $\mathcal{O}(2^{n})$ to $\mathcal{O}(2^{n-1})$.
* `parallel_probability(self, qubit: int) -> float`: A parallelized version of probability, potentially faster for large numbers of qubits. Uses the multiprocessing pool initialized in the constructor. `Note: It takes longer than using the single thread version 😂.`
* `measure(self, qubits: Union[List[int], int], rand: Callable[[], float] = None) -> list`: Simulates measuring the specified qubit(s). Returns a list of outcomes (0 or 1). Collapses the state vector. Optionally accepts a custom random number generating function rand (which should return floats between 0.0 and 1.0). Time complexity is $\mathcal{O}(m·2^{n})$ (where m is the number of qubits to measure).
* `parallel_measure(self, qubits: Union[List[int], int], rand: Callable[[], float] = None) -> list`: A parallelized version of measure. Uses parallel_probability and a parallelized collapse mechanism. Collapses the state vector.  `Note: Surprisingly, this actually takes less time than the single thread version 😲.`
* `density_matrix(self) -> np.array`: Calculates and returns the density matrix ρ = |ψ⟩⟨ψ| for the current state vector |ψ⟩. Returns a NumPy array of shape (2<sup>n_qubits</sup>, 2<sup>n_qubits</sup>).
* `print_probabilities(self, until : int = None)`: Prints the probabilities of each state. Useful for debugging. Time complexity is $\mathcal{O}(2^{n})$.
* `bloch_angles(self) -> tuple[float, float]`: For single-qubit systems only (n_qubits=1), calculates the polar angle θ and azimuthal angle φ representing the state on the Bloch sphere. Returns (theta, phi). Time complexity is $\mathcal{O}(1)$.
* `draw_bloch_sphere(self, filename: str = None)`: Draws the Bloch sphere representation of the single qubit state using matplotlib. If a filename is provided, it saves the figure image to a file. Time complexity is $\mathcal{O}(1)$.

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

# QDensity Usage

The `QDensity` class provides an alternative way to simulate quantum systems using the **density matrix formalism**. Unlike the state vector approach (`QRegistry`), the density matrix can represent not only pure quantum states but also **mixed states**, which are statistical ensembles of pure states. This is particularly useful for describing subsystems of larger entangled systems or simulating noise on quantum systems interacting with an environment.

The `QDensity` class allows you to:

*   Initialize a quantum system with a specified number of qubits, starting either in a pure state (e.g., |0...0⟩⟨0...0|) or potentially a mixed state (if initialized with a density matrix).
*   Retrieve the current density matrix (ρ).
*   Apply quantum gates to evolve the density matrix (ρ' = U ρ U<sup>†</sup>).
*   Calculate the probability of measurement outcomes for specific qubits.
*   Simulate the measurement of qubits, updating the density matrix according to the measurement outcome.
*   Perform a **partial trace** operation to obtain the reduced density matrix of a subsystem.
*   Calculate the purity of the state (Tr(ρ<sup>2</sup>)) to check if it's pure or mixed.

## Getting Started

### 1. Import Necessary Classes

```python
import numpy as np
from qdensity import QDensity # Assuming the class is in qdensity.py
from gates import Gate
```

### 2. Initialize the Density Matrix

Create an instance of `QDensity` by specifying the number of qubits. By default, it initializes to the pure state |0...0⟩⟨0...0|.

```python
# Create a 3-qubit density matrix registry
num_qubits = 3
density = QDensity(num_qubits)

# Output will be a 2^n x 2^n matrix with 1.0 at [0, 0] and 0.0 elsewhere
print("Initial Density Matrix:\n", density.get_density_matrix())
```

### 3. Apply Gates

Use the `apply_gates` method, similar to `QRegistry`. The method applies the unitary U corresponding to the gate(s) to the density matrix `ρ` using the formula ρ' = U ρ U<sup>†</sup>.

* **Single Gate Application:**
```python
# Apply Hadamard gate to qubit 0
density.apply_gates(Gate.H(), 0)

# Apply Pauli-X gate to qubit 1
density.apply_gates(Gate.X(), 1)

print("Density Matrix after H(0) and X(1):\n", density.get_density_matrix())
```

* **Multi-Qubit Gate Application (e.g., CNOT):** Works the same way as `QRegistry`, applying the gate's unitary `U` and its conjugate transpose U<sup>†</sup>.

```python
# Apply CNOT gate with qubit 0 as control and qubit 1 as target
density.apply_gates(Gate.CNOT(), 1)

print("Density Matrix after CNOT(0, 1):\n", density.get_density_matrix())
```

* **Applying Multiple Gates Simultaneously:**
```python
# Apply H to qubit 0 and X to qubit 2 simultaneously
density.apply_gates([Gate.H(), Gate.X()], [0, 2])

print("Density Matrix after H(0) and X(2):\n", density.get_density_matrix())
```

### 4. Partial Trace

A key operation for density matrices is the partial trace, which allows you to "discard" or "ignore" certain qubits and find the effective state of the remaining subsystem.

```python
# Create a 2-qubit Bell state |Φ+⟩ = (|00⟩ + |11⟩) / sqrt(2)
density = QDensity(2)
density.apply_gates(Gate.H(), 0)
density.apply_gates(Gate.CNOT(), 1)
print("Density Matrix (Bell State):\n", density.get_density_matrix())

# Trace out qubit 1 to get the reduced density matrix for qubit 0
# Provide the qubit to discard
reduced_density_q0 = density.partial_trace(1)
print("\nReduced Density Matrix for Qubit 0:\n", reduced_density_q0)
# Expected: [[0.5, 0], [0, 0.5]] (Maximally mixed state)

# Trace out qubit 0 to get the reduced density matrix for qubit 1
reduced_density_q1 = density.partial_trace(0)
print("\nReduced Density Matrix for Qubit 1:\n", reduced_density_q1)
# Expected: [[0.5, 0], [0, 0.5]] (Maximally mixed state)
```

### 5. Measure Qubits

Simulate measurement using the `measure` method. It calculates the outcome probabilities based on the diagonal elements of the density matrix (after applying appropriate projectors) and updates the density matrix based on the measurement outcome.

```python
# Continue with the Bell state example
print("Density Matrix before measurement:\n", density.get_density_matrix())

# Measure qubit 0
measurement_result_q0 = density.measure([0]) # Measure only qubit 0
print("\nMeasurement result for qubit 0:", measurement_result_q0)

# The density matrix is now collapsed based on the outcome for qubit 0
print("Density Matrix after measuring qubit 0:\n", density.get_density_matrix())
# If outcome was 0, state is |00⟩⟨00|
# If outcome was 1, state is |11⟩⟨11|

# Now measure qubit 1
measurement_result_q1 = density.measure([1])
print("\nMeasurement result for qubit 1:", measurement_result_q1)
print("Density Matrix after measuring qubit 1:\n", density.get_density_matrix())
# The final state will be |00⟩⟨00| or |11⟩⟨11| depending on the first measurement
```

### 6. Calculate Probabilities

Calculate the probability of measuring a specific qubit in the |1⟩ state without collapsing the state. This is typically done by calculating Tr( P<sub>1</sub> ρ ), where P<sub>1</sub> is the projector onto the |1⟩ state for that qubit.

```python
# Initialize a 1-qubit registry in state |+⟩
density = QDensity(1)
density.apply_gates(Gate.H(), 0)

# Calculate the probability of measuring qubit 0 as '1'
prob_1 = density.probability(0)
print(f"Probability of measuring |1⟩ on qubit 0: {prob_1:.4f}") # Should be ~0.5
```

## Key Methods Reference

_(Note: Complexity estimates assume dense density matrices of size N x N where N = 2<sup>n_qubits</sup>)._

* `__init__(self, n_qubits: int, initial_state=None)`: Constructor. Initializes a registry with `n_qubits`. Defaults to |0...0⟩⟨0...0|. Can accept a list of noise probabilities for each state. Time complexity: $\mathcal{O}(4^{n})$ to initialize the matrix.
* `get_density_matrix(self) -> np.array`: Returns the current density matrix as a complex array of 2<sup>n</sup> x 2<sup>n</sup>. Time complexity: $\mathcal{O}(1)$.
* `apply_gates(self, gates: Union[List[Gate], Gate], targets: Union[List[int], int])`: Applies one or more gates to the density matrix ρ via ρ' = U ρ U<sup>†</sup>. Modifies the internal density matrix. Time complexity: $\mathcal{O}(4^{n})$ Dominated by matrix multiplication.
* `partial_trace(self, keep_qubits: List[int]) -> np.array`: Calculates the reduced density matrix for the subsystem defined by `keep_qubits`. Returns a new, smaller density matrix. Uses Einstein`s Summation Notation to reduce the sub-matrices affected by the specified qubit.
* `probability(self, qubit: int) -> float`: Calculates the probability P(1) for the given qubit index by tracing over the appropriate projector and the density matrix. Does not modify the state. Time complexity: $\mathcal{O}(2^{n-1})$.
* `measure(self, qubits: Union[List[int], int], rand: Callable[[], float] = None) -> list`: Simulates measuring the specified qubit(s). Returns a list of outcomes (0 or 1). Collapses the internal density matrix according to the outcome. Time complexity: $\mathcal{O}(m · 2^{n})$ where m is the number of qubits to be meassured.
* `calculate_entropy(self) -> float`: Calculates the entropy of the system. Returns a float between $\epsilon$ (maximally mixed) and 0 (pure / no entropy).

## Notes on QDensity vs QRegistry

* `QDensity` uses matrices of size 2<sup>n</sup> x 2<sup>n</sup>, while `QRegistry` uses vectors of size 2<sup>n</sup>. This means `QDensity` requires significantly more memory ($\mathcal{O}(4^{n})$ vs $\mathcal{O}(2^{n})$).
* Operations on density matrices (especially gate application involving matrix multiplication) are generally more computationally expensive than corresponding state vector operations (often $\mathcal{O}(4^{n})$ for `QDensity` vs $\mathcal{O}(2^{n})$ for `QRegistry`).
* The main advantage of `QDensity` is its ability to represent mixed states, which is essential for describing subsystems or noisy quantum processes. `QRegistry` can only represent pure states.
* Partial trace is a fundamental operation unique to the density matrix formalism, though it can be also calculated on the `QRegistry` class after generating its density matrix.


## License

This project is licensed under the MIT License. Because every project needs to have a licence 😉.