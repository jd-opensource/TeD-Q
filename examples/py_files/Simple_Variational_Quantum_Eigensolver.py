import sys 
sys.path.append("..") 
import JDQAI as qai

# Related package
import torch
import numpy as np
import matplotlib.pyplot as plt

# Global variable
n_qubits = 4


"""
Creates a*I + b*Z + c*X + d*Y pauli sum 
that will be our Hamiltonian operator.

"""
weight = torch.tensor([1., 3., 2., 0.])

### Solve Hamiltonian by analytical method

from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver

def hamiltonian_operator(a, b, c, d):
    """
    Creates a*I + b*Z + c*X + d*Y pauli sum 
    that will be our Hamiltonian operator.
    
    """
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": a}, "label": "I"},
                   {"coeff": {"imag": 0.0, "real": b}, "label": "Z"},
                   {"coeff": {"imag": 0.0, "real": c}, "label": "X"},
                   {"coeff": {"imag": 0.0, "real": d}, "label": "Y"}
                   ]
    }
    return WeightedPauliOperator.from_dict(pauli_dict)
H = hamiltonian_operator(weight[0], weight[1], weight[2], weight[3])
exact_result = NumPyEigensolver(H).run()
reference_energy = min(np.real(exact_result.eigenvalues))
print('The exact ground state energy is: {}'.format(reference_energy))


### Define the circuit

def ansatz(theta):
    for j in range(n_qubits):
        qai.RY(theta[j], qubits=[j])

def circuitDef(theta):
    ansatz(theta)
    
    qai.Hadamard(qubits=[2])
    qai.S(qubits=[3])
    qai.Hadamard(qubits=[3])
    
    qai.measurement.expval(qai.I(qubits=[0]))
    qai.measurement.expval(qai.PauliZ(qubits=[1]))
    qai.measurement.expval(qai.PauliZ(qubits=[2]))
    qai.measurement.expval(qai.PauliZ(qubits=[3]))

circuit = qai.Circuit(circuitDef, n_qubits, torch.zeros(n_qubits))
compiledCircuit = circuit.compilecircuit('pytorch')

drawer = qai.matplotlib_drawer(circuit)
drawer.full_draw()



def cost(*params):
    x = params[0]
    exp_val = compiledCircuit(x*torch.ones([n_qubits], requires_grad=True))
    return torch.dot(weight,exp_val)


Optimizer = qai.GradientDescentOptimizer(cost, [0], 0.1, interface="pytorch")
init_value = torch.tensor(0.3, requires_grad=True)



new_params = (init_value)

opt_t_list = np.array([])
opt_c_list = np.array([])

opt_t_list = np.append(opt_t_list, new_params.item())
opt_c_list = np.append(opt_c_list, cost(new_params).item())
for i in range(100):
    new_params = Optimizer.step(new_params)[0]
    if (i + 1) % 5 == 0:
        opt_t_list = np.append(opt_t_list, new_params.item())
        opt_c_list = np.append(opt_c_list, cost(new_params).item())
    if (i + 1) % 20 == 0:
        
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(new_params)))
        print("Parameters after step {:5d}: {}".format(i + 1, new_params))
computed_energy = cost((new_params))



t_list = np.linspace(0, 2*np.pi, 101)
c_list = np.array([])
for t in t_list:
    c_list = np.append(c_list, cost(t).item())

plt.figure()
plt.plot(t_list, c_list)
plt.plot((opt_t_list)%(2*np.pi), opt_c_list)
plt.xlabel('Theta')
plt.ylabel('Energy')
plt.show()

print('The exact ground state energy is: \t{: .7f}'.format(reference_energy))
print('The computed ground state energy is: \t{: .7f}'.format(computed_energy))
print('Error is {: .5f}%'.format((computed_energy-reference_energy)/computed_energy*100))

