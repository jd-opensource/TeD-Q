import JDQAI as qai


# Define the quantum model
### Define the circuit with JDQAI framework

def circuitDef(*params):
    qai.RX(params[0], qubits=[0])
    qai.RY(params[1], qubits=[0])
    #return [qai.state()]
    return qai.expval(qai.PauliZ(qubits=[0]))

circuit = qai.Circuit(circuitDef, 1, 0.54, 0.12)

# Circuit compiled with JAX backend
hyper_opt = {'methods':['kahypar'], 'max_time':120, 'max_repeats':128, 'progbar':True, 'minimize':'flops', 'search_parallel':True, 'slicing_opts':{'target_size':2**28, 'repeats':128, 'target_num_slices':1, 'contract_parallel':False}}
my_compilecircuit = circuit.compilecircuit(backend="jax", use_jdopttn=True, hyper_opt = hyper_opt)

def cost(*params):
    return my_compilecircuit(*params)[0]

### Define cost function and optimizer
new_params = (0.54, 0.12)
cost(*new_params)

Optimizer = qai.GradientDescentOptimizer(cost, [0, 1], 0.4, interface="jax")

new_params = (0.011, 0.012)
for i in range(100):
    new_params = Optimizer.step(*new_params)
    #if (i + 1) % 5 == 0:
    #    print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(*new_params)))
    #    print("Parameters after step {:5d}: {}".format(i + 1, new_params))
print(new_params)
print(cost(*new_params))


# Circuit compiled with JAX backend and pytorch interface
my_compilecircuit = circuit.compilecircuit(backend="jax", interface="pytorch", use_cyc=False)

def cost(*params):
    return my_compilecircuit(*params)
Optimizer = qai.GradientDescentOptimizer(cost, [0, 1], 0.4, interface="pytorch")

import torch
a = torch.tensor([0.011], requires_grad= True)
b = torch.tensor([0.012], requires_grad= True)
my_params = (a, b)
ccc = cost(*my_params)


new_params = my_params
for i in range(100):
    new_params = Optimizer.step(*new_params)
    #if (i + 1) % 5 == 0:
    #    print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(*new_params)))
    #    print("Parameters after step {:5d}: {}".format(i + 1, new_params))
print(new_params)
print(cost(*new_params))




# Circuit compiled with pytorch backend

#import JDQAI as qai

def circuitDef(*params):
    qai.RX(params[0], qubits=[0])
    qai.RY(params[1], qubits=[0])
    qai.PauliX(qubits=[1])
    qai.PauliY(qubits=[2])
    qai.PauliZ(qubits=[3])
    qai.S(qubits=[4])
    qai.T(qubits=[5])
    qai.CNOT(qubits=[6, 7])
    qai.CY(qubits=[8, 9])
    qai.CZ(qubits=[10, 11])
    qai.SWAP(qubits=[12, 13])
    qai.CSWAP(qubits=[14, 15, 16])
    qai.Toffoli(qubits=[17, 18, 19])
    return qai.expval(qai.PauliZ(qubits=[0]))

circuit = qai.Circuit(circuitDef, 20, 0.54, 0.12)

hyper_opt = {'methods':['kahypar'], 'max_time':120, 'max_repeats':128, 'progbar':True, 'minimize':'flops', 'search_parallel':True, 'slicing_opts':{'target_size':2**28, 'repeats':128, 'target_num_slices':1, 'contract_parallel':True}}
my_compilecircuit = circuit.compilecircuit(backend="pytorch", use_jdopttn=True, hyper_opt = hyper_opt)

def cost(*params):
    return my_compilecircuit(*params)[0]

Optimizer = qai.GradientDescentOptimizer(cost, [0, 1], 0.4, interface="pytorch")

import torch

#device = torch.device('cuda:0')
device = torch.device('cpu')

a = torch.tensor([0.11], requires_grad= True, device = device)
b = torch.tensor([0.12], requires_grad= True, device = device)


#a = a.to(device)
#b = b.to(device)

my_params = (a, b)

c = cost(*my_params)
print(c)
c.backward()
print(a.grad, b.grad)

new_params = my_params
for i in range(100):
    new_params = Optimizer.step(*new_params)

print("Optimized rotation angles: {}".format(new_params))
print("Cost: {}".format(cost(*new_params)))