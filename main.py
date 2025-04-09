from multiprocessing import Pool
import time
from gates import Gate
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

if __name__ == '__main__':

    test_multiprocessing()

