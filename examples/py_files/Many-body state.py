import torch
import JDQAI as qai
import numpy as np

n_size = 4
n_qubits= n_size*n_size
t_d = 200e-9 #ns
g = 1
h_bar = 1


def getIndex(i, j):
    return n_size*i+j
def Hd(idx, jdx, di):
    qai.RZ(2*h_bar*di*t_d, qubits=[idx])
    
    H0(idx, jdx)
    
def H0(idx, jdx):
    qai.Hadamard(qubits=[idx])
    qai.Hadamard(qubits=[jdx])
    qai.CNOT(qubits=[jdx, idx])
    qai.RZ(torch.Tensor([g*h_bar*t_d]), qubits=[jdx], is_preparation=True, trainable_params=[])
    qai.CNOT(qubits=[jdx, idx])
    qai.Hadamard(qubits=[idx])
    qai.Hadamard(qubits=[jdx])
    
    qai.S(qubits=[idx])
    qai.S(qubits=[jdx])
    qai.Hadamard(qubits=[idx])
    qai.Hadamard(qubits=[jdx])
    qai.CNOT(qubits=[jdx, idx])
    qai.RZ(torch.Tensor([g*h_bar*t_d]), qubits=[jdx], is_preparation=True, trainable_params=[])
    qai.CNOT(qubits=[jdx, idx])
    qai.Hadamard(qubits=[idx])
    qai.Hadamard(qubits=[jdx])
    qai.S(qubits=[idx]).adjoint()
    qai.S(qubits=[jdx]).adjoint()
    
def circuitDef(theta, phi):
    # Neel state
    for i in range(n_size):
        for j in range(n_size):
            if (i+j)%2==0:
                qai.PauliX(qubits=[getIndex(i,j)])
    # Hd td
    for i in range(n_size):
        for j in range(n_size):
            if i+1>=0 and i+1<n_size:
                Hd(getIndex(i+1, j), getIndex(i,j), (np.random.rand()*2-1)*h_bar*2*np.pi)
            if i-1>=0 and i-1<n_size:
                Hd(getIndex(i-1, j), getIndex(i,j), (np.random.rand()*2-1)*h_bar*2*np.pi)
            if j+1>=0 and j+1<n_size:
                Hd(getIndex(i, j+1), getIndex(i,j), (np.random.rand()*2-1)*h_bar*2*np.pi)
            if j-1>=0 and j-1<n_size:
                Hd(getIndex(i, j-1), getIndex(i,j), (np.random.rand()*2-1)*h_bar*2*np.pi)
                
    # Trainable theta and phi
    for i in range(n_qubits):
        qai.RX(theta[i], qubits=[i]);
        qai.RZ(phi[i], qubits=[i]);
    # H0 dt     
    for i in range(n_size):
        for j in range(n_size):
            if i+1>=0 and i+1<n_size:
                H0(getIndex(i+1, j), getIndex(i,j))
            if i-1>=0 and i-1<n_size:
                H0(getIndex(i-1, j), getIndex(i,j))
            if j+1>=0 and j+1<n_size:
                H0(getIndex(i, j+1), getIndex(i,j))
            if j-1>=0 and j-1<n_size:
                H0(getIndex(i, j-1), getIndex(i,j))
    
    # Last rotation    
    qai.RX(theta[-1], qubits=[0])
    qai.RZ(phi[-1], qubits=[0])
    
#     qai.measurement.expval(qai.PauliZ(qubits=[0]))
    qai.measurement.probs(qubits=[0])
    


circuit = qai.Circuit(circuitDef, n_qubits, torch.ones(n_qubits+1), torch.ones(n_qubits+1))

hyper_opt = {'methods':['kahypar'], 'max_time':120, 'max_repeats':528, 'progbar':True, 'minimize':'flops', 'search_parallel':True, 'slicing_opts':{'target_size':2**26, 'repeats':328, 'target_num_slices':0, 'contract_parallel':True}}
my_compilecircuit = circuit.compilecircuit(backend="pytorch", use_jdopttn=True, hyper_opt = hyper_opt)


init_params = np.random.rand((n_qubits*2+1),2)*2-1
params = torch.rand(n_qubits*2+1,2, requires_grad=True)

d_erg = (np.random.rand(10, n_qubits)*2-1)*1*2*np.pi
d_local = (np.random.rand(10, n_qubits)*2-1)*40*2*np.pi

import torch.nn as nn
n_train = 100
loss = nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD([params], lr=0.01)

for epoch in range(n_train):
    y_input = torch.zeros(20)
    y_target = torch.zeros(20)
# for ergodic distribution
    for i in range(10):
        y = my_compilecircuit(torch.Tensor(d_erg[i]), params)
        y_h = torch.round(y[0][1])
        print("ergodic  ", i, y_h)
        y_input[i] = y[0][1]
        y_target[i] = 1
# for localized distribution
    for i in range(10):
        y = my_compilecircuit(torch.Tensor(d_local[i]), params)
        y_h = torch.round(y[0][1])
        print("localized  ", i, y_h)
        y_input[i+10] = y[0][1]
        y_target[i+10] = 0

    l = loss(y_input, y_target)
    l.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: loss = {l:.8f}')
