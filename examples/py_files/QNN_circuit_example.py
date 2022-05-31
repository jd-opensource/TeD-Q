r"""
QNN circuit example


"""
# Import JDQAI
import sys 
sys.path.append("..") 
import JDQAI as qai

# Related package
import torch
import numpy as np
import matplotlib.pyplot as plt

# Global variable
n_qubits = 6
depth=1

### Fully connected layer
rand_params = np.random.uniform(high=2 * np.pi, size=((depth+1)*3, n_qubits))
def circuitDef(theta):
    for idx, element in enumerate(theta):
        qai.RY(element, qubits=[idx], is_preparation=True)
    qai.Templates.FullyConnected(n_qubits, depth, rand_params)
    
    exp_vals = [qai.measurement.expval(qai.PauliZ(qubits=[position])) for position in range(n_qubits)] 

# Build the circuit

circuit = qai.Circuit(circuitDef, n_qubits, torch.zeros(n_qubits))
compiledCircuit = circuit.compilecircuit('pytorch')

# Visualize the circuit

drawer = qai.matplotlib_drawer(circuit)
drawer.full_draw()
plt.show()



### Hardware Efficient circuit

rand_params = np.random.uniform(high=2 * np.pi, size=((depth+1)*3, n_qubits))
def circuitDef(theta):
    for idx, element in enumerate(theta):
        qai.RY(element, qubits=[idx], is_preparation=True)
    qai.Templates.HardwareEfficient(n_qubits, depth, rand_params)
    
    exp_vals = [qai.measurement.expval(qai.PauliZ(qubits=[position])) for position in range(n_qubits)]

# Build the circuit

circuit = qai.Circuit(circuitDef, n_qubits, torch.zeros(n_qubits))
compiledCircuit = circuit.compilecircuit('pytorch')

# Visualize the circuit

drawer = qai.matplotlib_drawer(circuit)
drawer.full_draw()