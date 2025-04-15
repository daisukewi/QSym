
from gates import Gate
from helper import prob_state_reg
from qregistry import QRegistry


def deutsch_jozsa():
    q_qubits = 4
    r = QRegistry(q_qubits + 1)

    r.apply_gates(Gate.H(), [0, 1, 2, 3])
    r.apply_gates(Gate.X(), 4)
    r.apply_gates(Gate.H(), 4)
    

    # Apply the oracle for balanced function f(x)
    r.apply_gates(Gate.X(), [0, 3])
    r.apply_c_gate(Gate.CNOT(), 4, 0)
    r.apply_c_gate(Gate.CNOT(), 4, 1)
    r.apply_c_gate(Gate.CNOT(), 4, 2)
    r.apply_c_gate(Gate.CNOT(), 4, 3)
    r.apply_gates(Gate.X(), [0, 3])

    # Apply the Deutsch-Jozsa algorithm to the first 4 qubits
    r.apply_gates(Gate.H(), [0, 1, 2, 3])

    r.print_probabilities(16)

    # Measure the first 4 qubits
    result = r.measure([0, 1, 2, 3])
    print(f"Deutsch-Jozsa Result: {result}")

    # Check if the result is all zeros
    if all(bit == 0 for bit in result):
        print("The function is constant.")
    else:
        print("The function is balanced.")

    return

if __name__ == '__main__':

    deutsch_jozsa()