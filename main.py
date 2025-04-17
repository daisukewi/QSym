import time
from gates import Gate
import numpy as np
from helper import calculate_entropy
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
    # Bloch Sphere Angles
    # Link to check the results:
    # https://algassert.com/quirk#circuit={%22cols%22:[[1,%22X%22,%22H%22,%22Z%22,%22X^%C2%BC%22,%22Y^%C2%BC%22],[1,1,1,%22H%22]]}
    
    r = QRegistry(1)
    phi, theta = r.bloch_angles()
    print(f"Bloch Sphere |0> φ={phi*180/np.pi:.0f}º , θ={theta*180/np.pi:.0f}º\n")

    r.apply_gates(Gate.X(), 0)
    phi, theta = r.bloch_angles()
    print(f"Bloch Sphere |1> φ={phi*180/np.pi:.0f}º , θ={theta*180/np.pi:.0f}º\n")

    r = QRegistry(1)
    r.apply_gates(Gate.H(), 0)
    phi, theta = r.bloch_angles()
    print(f"Bloch Sphere |+> φ={phi*180/np.pi:.0f}º , θ={theta*180/np.pi:.0f}º\n")

    r = QRegistry(1)
    r.apply_gates(Gate.Z(), 0)
    r.apply_gates(Gate.H(), 0)
    phi, theta = r.bloch_angles()
    print(f"Bloch Sphere |-> φ={phi*180/np.pi:.0f}º , θ={theta*180/np.pi:.0f}º\n")

    r = QRegistry(1)
    r.apply_gates(Gate.RX(np.pi/4), 0)
    phi, theta = r.bloch_angles()
    print(f"Bloch Sphere |ψ⟩ φ={phi*180/np.pi:.0f}º , θ={theta*180/np.pi:.0f}º\n")

    r = QRegistry(1)
    r.apply_gates(Gate.RY(np.pi/4), 0)
    phi, theta = r.bloch_angles()
    print(f"Bloch Sphere |ψ⟩ φ={phi*180/np.pi:.0f}º , θ={theta*180/np.pi:.0f}º\n")

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
    d = QDensity(3)
    d.apply_gates(Gate.H(), 0)
    d.apply_gates(Gate.CNOT(), 1)

    print("rho:\n", d.get_density)
    print("\nTrace of rho:", np.trace(d.get_density))
    tr0 = d.partial_trace(0)
    print("Partial Trace rho - Q0:\n", tr0)
    print("Entropy rho:", d.calculate_entropy())
    print("Entropy rho - Q0:", calculate_entropy(tr0))

if __name__ == '__main__':

    test_angles()

