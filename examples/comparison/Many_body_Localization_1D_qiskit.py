import numpy as np
from qiskit import *
import torch
import math
from torch.optim import lr_scheduler
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.utils import QuantumInstance, algorithm_globals

import sys
import time

import warnings
warnings.filterwarnings('ignore')



n_size = int(sys.argv[1])
u_device = sys.argv[2]

n_epoch = 50
n_train = 40
n_test = 20


lr = 0.01
gamma_lr_scheduler = 0.2

n_qubits= n_size
t_d = 200e-9 #ns
h_bar = 1
g = 2.185e6 #MHz
h_erg = 1e6 # MHZ
h_loc = 40e6  # MHz


def getIndex(i, j):
    return n_size*i+j
def Hd(circ, idx, jdx, di):
    #qai.RZ(2*h_bar*di*t_d, qubits=[jdx], trainable_params=[])
    circ.rz(di, jdx)
    H0(circ, idx, jdx)
    
def H0(circ, idx, jdx):
    circ.h(idx)
    circ.h(jdx)
    circ.cx(idx, jdx)
    circ.rz(g*h_bar*t_d,jdx)
    circ.cx(idx, jdx)
    circ.h(idx)
    circ.h(jdx)
    
    circ.s(idx)
    circ.s(jdx)
    circ.h(idx)
    circ.h(jdx)
    circ.cx(idx, jdx)
    circ.rz(g*h_bar*t_d, jdx)
    circ.cx(idx, jdx)
    circ.h(idx)
    circ.h(jdx)
    # qai.PhaseShift(torch.tensor(-math.pi/2.), qubits=[idx], trainable_params=[])
    # qai.PhaseShift(torch.tensor(-math.pi/2.), qubits=[jdx], trainable_params=[])
    circ.sdg(idx)# add adjoint here will not change circuit's operator
    circ.sdg(jdx)
    
N = (n_size-1)*2

d_list = []
for idx in range(N):
    d_list.append(Parameter("d_"+str(idx)))

n_train_params = (n_qubits+1)*2
params_list = []
for idx in range(n_train_params):
    params_list.append(Parameter("param_"+str(idx)))

#     params = params.view(-1)
preparation_circ = QuantumCircuit(n_qubits, 0)
qnn_circ = QuantumCircuit(n_qubits, 0)
    
    # Neel state
for i in range(n_qubits):
    if i%2==0:
        preparation_circ.x(i)
    
    
# Hd td
count = 0
for i in range(n_qubits):
    if i>=0 and i+1<n_qubits:# (n_size-1)*n_size
        Hd(preparation_circ, i+1, i, d_list[count])
        count+=1
    if i-1>=0 and i<n_qubits:# (n_size-1)*n_size
        Hd(preparation_circ, i-1, i, d_list[count])
        count+=1
#print("count:  ", count)
    
            
# Trainable theta and phi
for i in range(n_qubits):
    qnn_circ.ry(params_list[2*i], i)
    qnn_circ.rx(params_list[2*i+1], i)
    qnn_circ.ry(-params_list[2*i], i)
    
    # H0 dt 
    
    
for i in range(n_size):
    if i>=0 and i+1<n_qubits:# (n_size-1)*n_size
        H0(qnn_circ,i+1, i)
    if i-1>=0 and i<n_qubits:# (n_size-1)*n_size
        H0(qnn_circ,i-1, i)
    
    
# Last rotation  
qnn_circ.ry(params_list[2*n_qubits], 0)
qnn_circ.rx(params_list[2*n_qubits+1], 0)
qnn_circ.ry(-params_list[2*n_qubits], 0)
    #RZ
    
#     qai.measurement.expval(qai.PauliZ(qubits=[0]))
    # qai.measurement.probs(qubits=[0])
    
qc = QuantumCircuit(n_qubits, 0)
qc.append(preparation_circ, range(n_qubits))
qc.append(qnn_circ, range(n_qubits))
#qc.measure(0,0)

# Define CircuitQNN and initial setup
parity = lambda x: "{:b}".format(x).count("1") % 2  # optional interpret function
output_shape = 2  # parity = 0, 1

if u_device == 'cpu':
    qi = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))
elif u_device == 'gpu':
    qi = QuantumInstance(Aer.get_backend("aer_simulator_statevector_gpu"))
else:
    raise ValueError("unknow device!!!")
#qi = QuantumInstance(Aer.get_backend("aer_simulator_matrix_product_state"))
#qi = QuantumInstance(Aer.get_backend("aer_simulator_density_matrix"))

qnn2 = CircuitQNN(
    qc,
    input_params=preparation_circ.parameters,
    weight_params=qnn_circ.parameters,
    interpret=parity,
    output_shape=output_shape,
    quantum_instance=qi,
)


initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn2.num_weights) - 1)
#print("Initial weights: ", initial_weights)

model2 = TorchConnector(qnn2, initial_weights)

model2 = TorchConnector(qnn2)

# prepare training data
params = torch.rand(n_qubits+1,2, requires_grad=True)


d_erg = torch.tensor(np.random.rand(int(n_train/2), N)*2-1)*h_erg*h_bar*t_d*math.pi
d_local = torch.tensor((np.random.rand(int(n_train/2), N)*39/40.+1/40.)*np.random.choice([-1., 1.], size=(int(n_train/2), N))*h_loc*h_bar*t_d*math.pi)
#d_local = torch.tensor(np.random.rand(int(n_train/2), N)*2-1)*h_loc*h_bar*t_d*math.pi
d = torch.cat((d_erg, d_local), 0)

y_target = torch.Tensor(np.array([1]*int(n_train/2)+[0]*int(n_train/2)))


optimizer = torch.optim.Adam(model2.parameters(), lr=0.5)



import torch.nn as nn
rnd_sq = np.arange(n_train)


target_list = [0 for _ in range(n_train)]
y_list = [0 for _ in range(n_train)]


time_start = time.time()

for epoch in range(n_epoch):
    np.random.shuffle(rnd_sq)
    l_sum = 0
    
    for i in rnd_sq:
        #w = y_target[i]*2+1
        loss = nn.BCELoss(reduction='mean')
        #cir_params = torch.cat((params, -params[:,0].view(-1,1)),1)
        #print(d[i])
        y = model2(d[i])
        
        #print(y)
        
        diff = y[1] - 0.6
        if diff > 0:
            diff = diff*5./4. + 0.5
        else:
            diff = diff*5./6. + 0.5
        
        diff = y[1]
        
        l = loss(diff, y_target[i])
        l.backward()
        #print(params.grad)
        
        l_sum = l_sum + l
        target_list[i] = y_target[i]
        y_list[i] = diff.data
    

    #if epoch % 5 == 0:
    #    print(f'epoch {epoch + 1}: loss = {l_sum/(n_train-5):.8f}')
    #    print("acc:", np.sum((np.round(y_list)==target_list))/n_train*100)
    #    print("prediction:  ", y_list[0:n_train//2], "   ", y_list[n_train//2:])
    #    #print("target:   ", target_list)
    #    #print(params.grad)
        
    next(model2.parameters()).grad /= n_train
    optimizer.step()
    optimizer.zero_grad()
    
    #exp_lr_scheduler.step()

time_end = time.time()

print(' ')
print(' ')
print('number of qubits: ', n_size)
print('device: ', u_device)
print('qiskit time cost for 50 epochs, 40 trains per epoch: ', time_end-time_start, 's')