import torch
import tedq as qai
import numpy as np
import math
from torch.optim import lr_scheduler
import sys
from jdtensorpath.distributed import run_distributed


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

if u_device == 'gpu0':
    device = torch.device("cuda:0")

elif u_device == 'gpu1':
    device = torch.device("cuda:1")

elif u_device == 'gpu2':
    device = torch.device("cuda:2")

elif u_device == 'gpu3':
    device = torch.device("cuda:3")

elif u_device == 'cpu':
    device = torch.device("cpu")
else:
    raise ValueError("unknow device")



def getIndex(i, j):
    return n_size*i+j
def Hd(idx, jdx, di):
    #qai.RZ(2*h_bar*di*t_d, qubits=[jdx], trainable_params=[])
    qai.RZ(di, qubits=[jdx])
    H0(idx, jdx)
    
def H0(idx, jdx):
    qai.Hadamard(qubits=[idx])
    qai.Hadamard(qubits=[jdx])
    qai.CNOT(qubits=[idx, jdx])
    qai.RZ(torch.tensor([g*h_bar*t_d], device=device), qubits=[jdx], trainable_params=[])
    qai.CNOT(qubits=[idx, jdx])
    qai.Hadamard(qubits=[idx])
    qai.Hadamard(qubits=[jdx])
    
    qai.S(qubits=[idx])
    qai.S(qubits=[jdx])
    qai.Hadamard(qubits=[idx])
    qai.Hadamard(qubits=[jdx])
    qai.CNOT(qubits=[idx, jdx])
    qai.RZ(torch.tensor([g*h_bar*t_d], device=device), qubits=[jdx], trainable_params=[])
    qai.CNOT(qubits=[idx, jdx])
    qai.Hadamard(qubits=[idx])
    qai.Hadamard(qubits=[jdx])
    qai.PhaseShift(torch.tensor(-math.pi/2., device=device), qubits=[idx], trainable_params=[])
    qai.PhaseShift(torch.tensor(-math.pi/2., device=device), qubits=[jdx], trainable_params=[])
    #qai.S(qubits=[idx]).adjoint()# add adjoint here will not change circuit's operator
    #qai.S(qubits=[jdx]).adjoint()

def partial_connected():
    for i in range(0, n_qubits-7, 5):
        #for j in range(i+1, n_qubits, 5):
        if i+7 < n_qubits:
            qai.CNOT(qubits=[i, i+7])


def circuitDef(d, params):
    
    params = params.view(-1)
    
    
    # Neel state
    for i in range(n_qubits):
        if i%2==0:
            qai.PauliX(qubits=[i])
    
    
    # Hd td
    count = 0
    for i in range(n_qubits):
        if i>=0 and i+1<n_qubits:# (n_size-1)*n_size
            Hd(i+1, i, d[count])
            count+=1
        if i-1>=0 and i<n_qubits:# (n_size-1)*n_size
            Hd(i-1, i, d[count])
            count+=1
    #print("count:  ", count)
    
            
    # Trainable theta and phi
    for i in range(n_qubits):
        #print(i)
        qai.RY(params[2*i], qubits=[i])
        qai.RX(params[2*i+1], qubits=[i])
        qai.RY(-params[2*i], qubits=[i])
    
    # H0 dt 
    
    partial_connected()
    
    for i in range(n_size):
        if i+1>=0 and i+1<n_qubits:# (n_size-1)*n_size
            H0(i+1, i)
        if i-1>=0 and i-1<n_qubits:# (n_size-1)*n_size
            H0(i-1, i)

                     
    
    
    
    # Last rotation  
    qai.RY(params[2*n_qubits], qubits=[n_qubits-1])
    qai.RX(params[2*n_qubits+1], qubits=[n_qubits-1])
    qai.RY(-params[2*n_qubits], qubits=[n_qubits-1])
    #RZ
    
#     qai.measurement.expval(qai.PauliZ(qubits=[0]))
    qai.measurement.probs(qubits=[n_qubits-1])
    
N = (n_size-1)*2

circuit = qai.Circuit(circuitDef, n_qubits, torch.rand(N, device=device), torch.rand(n_qubits+1,2, device=device))

#my_compilecircuit = circuit.compilecircuit(backend="pytorch" )

from jdtensorpath import JDOptTN as jdopttn
slicing_opts = None#{'target_size':2**28, 'repeats':100, 'target_num_slices':1, 'contract_parallel':False}
hyper_opt = {'methods':['kahypar'], 'max_time':120, 'max_repeats':56, 'progbar':True, 'minimize':'flops', 'search_parallel':True, 'slicing_opts':slicing_opts}
my_compilecircuit = circuit.compilecircuit(backend="pytorch", use_jdopttn=jdopttn, hyper_opt = hyper_opt, tn_simplify = False)

params = torch.rand(n_qubits+1,2, requires_grad=True, device=device)


d_erg = torch.tensor(np.random.rand(int(n_train/2), N)*2-1, device=device)*h_erg*h_bar*t_d*math.pi
d_local = torch.tensor((np.random.rand(int(n_train/2), N)*39/40.+1/40.)*np.random.choice([-1., 1.], size=(int(n_train/2), N))*h_loc*h_bar*t_d*math.pi, device=device)
#d_local = torch.tensor(np.random.rand(int(n_train/2), N)*2-1)*h_loc*h_bar*t_d*math.pi
d = torch.cat((d_erg, d_local), 0)

y_target = torch.Tensor(np.array([1]*int(n_train/2)+[0]*int(n_train/2)))
y_target = y_target.to(device)



import torch.nn as nn




optimizer = torch.optim.Adam([params], lr=0.5)

rnd_sq = np.arange(n_train)


target_list = [0 for _ in range(n_train)]
y_list = [0 for _ in range(n_train)]




def loss_function(_d, params):
    
    loss = nn.BCELoss(reduction='mean')
    cir_params = torch.cat((params, -params[:,0].view(-1,1)),1)
    #print(_d)
    #print(cir_params)
    y = my_compilecircuit(_d, cir_params)
    #print(y)
        
    diff = y[0][1] - 0.6
    if diff > 0:
        diff = diff*5./4. + 0.5
    else:
        diff = diff*5./6. + 0.5
        
    diff = y[0][1]
        
    #print(y[0][1], diff, y_target[i])
    l = loss(diff, y_target[i])
    
    return (l, diff)

run = run_distributed(2, 0)
run.set_circuit(loss_function, backward_index=0)

time_start = time.time()

for epoch in range(n_epoch):
    np.random.shuffle(rnd_sq)
    l_sum = 0
    
    for i in rnd_sq:

        (l, y_list[i]) = run(d[i], params)

        l_sum = l_sum + l
        target_list[i] = y_target[i]
    

        
    params.grad = params.grad/n_train
    #print(params.grad)
    optimizer.step()
    optimizer.zero_grad()

    y_list = [tensor.detach().numpy() for tensor in y_list]

    #if epoch % 5 == 0:
    #    print(f'epoch {epoch + 1}: loss = {l_sum/(n_train-5):.8f}')
    #    print("acc:", np.sum((np.round(y_list)==target_list))/n_train*100)
    #    print("prediction:  ", y_list[0:n_train//2], "   ", y_list[n_train//2:])
    #    #print("target:   ", target_list)
    #    #print(params.grad)    
    #exp_lr_scheduler.step()

time_end = time.time()


print(' ')
print(' ')
print('number of qubits: ', n_size)
print('device: ', 'distributed gpus, 5 nodes, 4 GPU per nodes')
print('TeD-Q distributed tensor network contraction mode time cost for 50 epochs, 40 trains per epoch: ', time_end-time_start, 's')
#state vector propagation
