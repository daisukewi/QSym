from multiprocessing import Pool
import time
from gates import Gate, partial_trace
import numpy as np
from qregistry import QRegistry
from qdensity import QDensity


def test_bell_states():
    # Bell States
    r = QRegistry(2)
    r.apply_gates(Gate.H(), 0)
    r.apply_gates(Gate.CNOT(), 1)
    print("\n∣Φ+⟩=1/sqrt(2)*​(∣00⟩+∣11⟩)")
    print("State:\n", r.get_state())
    print(r.measure([0, 1]))
    print("State:\n", r.get_state())

    r = QRegistry(2)
    r.apply_gates(Gate.X(), 0)
    r.apply_gates(Gate.H(), 0)
    r.apply_gates(Gate.CNOT(), 1)
    print("\n∣Φ−⟩=1/sqrt(2)*(∣00⟩−∣11⟩)")
    print("State:\n", r.get_state())
    print(r.measure([0, 1]))
    print("State:\n", r.get_state())

    r = QRegistry(2)
    r.apply_gates([Gate.H(), Gate.X()], [0, 1])
    r.apply_gates(Gate.CNOT(), 1)
    print("\n∣Ψ+⟩=1/sqrt(2)*​(∣01⟩+∣10⟩)")
    print("State:\n", r.get_state())
    print(r.measure([0, 1]))
    print("State:\n", r.get_state())

    r = QRegistry(2)
    r.apply_gates(Gate.X(), [0, 1])
    r.apply_gates(Gate.H(), 0)
    r.apply_gates(Gate.CNOT(), 1)
    print("\n∣Ψ−⟩=1/sqrt(2)*​(∣01⟩−∣10⟩)")
    print("State:\n", r.get_state())
    print(r.measure([0, 1]))
    print("State:\n", r.get_state())

    print("Density Matrix:", r.density_matrix())

def test_angles():
    r = QRegistry(1)
    phy, theta = r.bloch_angles()
    print(f"Bloch Sphere |0> φ = {phy:0.4f} - θ = {theta:0.4f}\n")

    r.apply_gates(Gate.X(), 0)
    phy, theta = r.bloch_angles()
    print(f"Bloch Sphere |1> φ = {phy:0.4f} - θ = {theta:0.4f}\n")

    r = QRegistry(1)
    r.apply_gates(Gate.H(), 0)
    theta, phi = r.bloch_angles()
    print(f"Bloch Sphere |+> φ = {phy:0.4f} - θ = {theta:0.4f}\n")

    r = QRegistry(1)
    r.apply_gates([Gate.Z(), Gate.H()], 0)
    theta, phi = r.bloch_angles()
    print(f"Bloch Sphere |-> φ = {phy:0.4f} - θ = {theta:0.4f}\n")

    r = QRegistry(1)
    r.apply_gates(Gate.RY(np.pi/3), 0)
    #r.apply_gates(Gate.RZ(np.pi/4), 0)
    phy, theta = r.bloch_angles()
    print(f"Bloch Sphere |ψ⟩ φ = {phy:0.4f} - θ = {theta:0.4f}\n")

def test_probabilities():
    # Probabilty Tests
    r = QRegistry(5)

    r.apply_gates(Gate.H(), 0)
    r.apply_gates(Gate.CNOT(), 1)
    r.apply_gates(Gate.CNOT(), 2)
    r.apply_gates(Gate.CNOT(), 3)
    r.apply_gates(Gate.CNOT(), 4)

    print("State:\n", r.get_state())
    print(f"Prob 1: {r.probability(0):0.2f}")
    print(f"Prob 2: {r.probability(1):0.2f}")
    print(f"Prob 3: {r.probability(2):0.2f}")
    print(f"Prob 4: {r.probability(3):0.2f}")
    print(f"Prob 5: {r.probability(4):0.2f}")

    print("Measure: ", r.measure([0, 1, 2, 3, 4]))

def test_multiprocessing():
    # Multiprocessing Tests
    time_start = time.time()
    r = QRegistry(25)
    print("Time Create Registry:", time.time() - time_start)

    time_start = time.time()
    r.apply_gates(Gate.H(), 0)
    r.apply_gates(Gate.CNOT(), 1)
    r.apply_gates(Gate.CNOT(), 2)
    r.apply_gates(Gate.CNOT(), 12)
    r.apply_gates(Gate.CNOT(), 24)
    print("Time Apply Gates:", time.time() - time_start)

    time_start = time.time()
    print(f"Prob 0: {r.parallel_probability(0):0.2f}")
    print("Time Prob Parallel:", time.time() - time_start)
    time_start = time.time()
    print(f"Prob 0: {r.probability(0):0.2f}")
    print("Time Prob Lineal:", time.time() - time_start)

    time_start = time.time()
    print(f"Prob 12: {r.parallel_probability(12):0.2f}")
    print("Time Prob Parallel:", time.time() - time_start)
    time_start = time.time()
    print(f"Prob 12: {r.probability(12):0.2f}")
    print("Time Prob Lineal:", time.time() - time_start)

    # time_start = time.time()
    # print("Measure 11,12,13:", r.parallel_measure([11, 12, 13]))
    # print("Time Measure Parallel:", time.time() - time_start)
    # time_start = time.time()
    # print("Measure 22,23,24:", r.measure([22, 23, 24]))
    # print("Time Measure Lineal:", time.time() - time_start)

def test_density():
    d = QDensity(2)
    print("Density Matrix:\n", d.get_density)
    d.apply_gates(Gate.X(), 0)
    d.apply_gates(Gate.H(), [0, 1])
    
    print("Density Matrix:\n", d.get_density)
    print()

    r = QRegistry(2)
    print("Density Matrix:\n", r.density_matrix())
    r.apply_gates(Gate.X(), 0)
    r.apply_gates(Gate.H(), [0, 1])
    
    print("Density Matrix:\n", r.density_matrix())

def test_density_with_noise():
    state_probabilities = [
        (0.1, 0),  # P(|000>)
        (0.0, 1),  # P(|001>)
        (0.2, 2),  # P(|010>)
        (0.1, 3),  # P(|011>)
        (0.05, 4), # P(|100>)
        (0.15, 5), # P(|101>)
        (0.3, 6),  # P(|110>)
        (0.1, 7)   # P(|111>)
    ]
    d = QDensity(3, state_probabilities)
    rho = d.get_density
    print("Density Matrix:\n", rho)
    print(f"\nTrace of rho: {np.trace(rho)}")

def test_density_probabilities():
    # Probabilty Tests
    d = QDensity(3)

    d.apply_gates(Gate.H(), 0)
    d.apply_gates(Gate.CNOT(), 1)

    print("rho:\n", d.get_density)
    print(f"\nProb 1: {d.probability(0):0.2f}")
    print(f"Prob 2: {d.probability(1):0.2f}")
    print(f"Prob 3: {d.probability(2):0.2f}")

    print("Measure: ", d.measure([0, 1, 2]))
    print("\nrho:\n", d.get_density)

def test_density_entropy():
    state_probabilities = [
        (0.1, 0),  # P(|000>)
        (0.0, 1),  # P(|001>)
        (0.2, 2),  # P(|010>)
        (0.1, 3),  # P(|011>)
        (0.05, 4), # P(|100>)
        (0.15, 5), # P(|101>)
        (0.3, 6),  # P(|110>)
        (0.1, 7)   # P(|111>)
    ]
    d = QDensity(3, state_probabilities)

    print("rho:\n", d.get_density)
    print("\nEntropy:", d.calculate_entropy())

    d.apply_gates(Gate.H(), 0)
    d.apply_gates(Gate.CNOT(), 1)
    print("rho:\n", d.get_density)
    print("\nEntropy ∣Φ+⟩:", d.calculate_entropy())

    d = QDensity(3, state_probabilities)
    d.apply_gates(Gate.X(), 0)
    d.apply_gates(Gate.H(), 0)
    d.apply_gates(Gate.CNOT(), 1)
    print("rho:\n", d.get_density)
    print("\nEntropy ∣Φ−⟩:", d.calculate_entropy())

    d = QDensity(3, state_probabilities)
    d.apply_gates([Gate.H(), Gate.X()], [0, 1])
    d.apply_gates(Gate.CNOT(), 1)
    print("rho:\n", d.get_density)
    print("\nEntropy ∣Ψ+⟩:", d.calculate_entropy())

    d = QDensity(3, state_probabilities)
    d.apply_gates(Gate.X(), [0, 1])
    d.apply_gates(Gate.H(), 0)
    d.apply_gates(Gate.CNOT(), 1)
    print("rho:\n", d.get_density)
    print("\nEntropy ∣Ψ-⟩:", d.calculate_entropy())

def test_density_partial_trace():
    # d = QDensity(3)
    # d.apply_gates(Gate.H(), [0, 2])
    # d.apply_gates(Gate.CNOT(), 1)

    # print("rho:\n", d.get_density)
    # print("\nTrace of rho:", np.trace(d.get_density))
    # print("Partial Trace rho:\n", d.partial_trace([1]))

    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30, 31, 32],
                  [33, 34, 35, 36, 37, 38, 39, 40], [41, 42, 43, 44, 45, 46, 47, 48], [49, 50, 51, 52, 53, 54, 55, 56], [57, 58, 59, 60, 61, 62, 63, 64]])
    print("partial trace:\n", partial_trace(x, 3, [2]))

if __name__ == '__main__':

    test_density_partial_trace()

