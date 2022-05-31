import pennylane as qai
import numpy as np
import math
import torch
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

dev1 = qai.device("default.qubit.torch", wires=n_qubits)

def getIndex(i, j):
    return n_size*i+j
def Hd(idx, jdx, di):
    qai.RZ(di, wires=jdx)
    H0(idx, jdx)
    
def H0(idx, jdx):
    qai.Hadamard(wires=idx)
    qai.Hadamard(wires=jdx)
    qai.CNOT(wires=[idx, jdx])
    qai.RZ(torch.tensor(g*h_bar*t_d, device=device), wires=jdx)
    qai.CNOT(wires=[idx, jdx])
    qai.Hadamard(wires=idx)
    qai.Hadamard(wires=jdx)
    
    qai.S(wires=idx)
    qai.S(wires=jdx)
    qai.Hadamard(wires=idx)
    qai.Hadamard(wires=jdx)
    qai.CNOT(wires=[idx, jdx])
    qai.RZ(torch.tensor(g*h_bar*t_d, device=device), wires=jdx)
    qai.CNOT(wires=[idx, jdx])
    qai.Hadamard(wires=idx)
    qai.Hadamard(wires=jdx)
    qai.PhaseShift(torch.tensor(-math.pi/2., device=device), wires=idx)
    qai.PhaseShift(torch.tensor(-math.pi/2., device=device), wires=jdx)
    #qai.S(qubits=[idx]).adjoint()# add adjoint here will not change circuit's operator
    #qai.S(qubits=[jdx]).adjoint()

def partial_connected():
    for i in range(0, n_qubits-7, 5):
        #for j in range(i+1, n_qubits, 5):
        if i+7 < n_qubits:
            qai.CNOT(wires=[i, i+7])
        
@qai.qnode(dev1, interface="torch")   
def circuitDef(d, params):
    
    params = params.view(-1)
    
    
    # Neel state
    for i in range(n_qubits):
        if i%2==0:
            qai.PauliX(wires=i)
    
    
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
        qai.RZ(params[2*i], wires=i)
        qai.RX(params[2*i+1], wires=i)
        qai.RZ(-params[2*i], wires=i)
    
    # H0 dt 
    
    partial_connected()
    
    '''
    for i in range(n_size):
        if i+1>=0 and i+1<n_qubits:# (n_size-1)*n_size
            H0(i+1, i)
        if i-1>=0 and i-1<n_qubits:# (n_size-1)*n_size
            H0(i-1, i)
    '''
    
    # Last rotation  
    qai.RZ(params[2*n_qubits], wires=n_qubits-1)
    qai.RX(params[2*n_qubits+1], wires=n_qubits-1)
    qai.RZ(-params[2*n_qubits], wires=n_qubits-1)
    #RZ
    
    #return qai.expval(qai.PauliZ(wires=[0]))
    return qai.probs(wires=n_qubits-1)
    

    
N = (n_size-1)*2

params = torch.rand(n_qubits+1,2, requires_grad=True, device=device)


d_erg = torch.tensor(np.random.rand(int(n_train/2), N)*2-1, device=device)*h_erg*h_bar*t_d*math.pi
d_local = torch.tensor((np.random.rand(int(n_train/2), N)*39/40.+1/40.)*np.random.choice([-1., 1.], size=(int(n_train/2), N))*h_loc*h_bar*t_d*math.pi, device=device)
#d_local = torch.tensor(np.random.rand(int(n_train/2), N)*2-1)*h_loc*h_bar*t_d*math.pi
d = torch.cat((d_erg, d_local), 0)

y_target = torch.Tensor(np.array([1]*int(n_train/2)+[0]*int(n_train/2)))
y_target = y_target.double()
y_target = y_target.to(device)


d = d.float()
params = params.float()



import torch.nn as nn


optimizer = torch.optim.Adam([params], lr=0.5)

rnd_sq = np.arange(n_train)


target_list = [0 for _ in range(n_train)]
y_list = [0 for _ in range(n_train)]

time_start = time.time()


for epoch in range(n_epoch):
    np.random.shuffle(rnd_sq)
    l_sum = 0
    
    for i in rnd_sq:
        w = y_target[i]*2+1
        loss = nn.BCELoss(reduction='mean')
        #cir_params = torch.cat((params, -params[:,0].view(-1,1)),1)
        dd = d[i]
        #print(dd)
        #print(params)
        #n_params = params.view(-1)
        y = circuitDef(dd, params)
        #print(y)
        #print(y[1])
        
        diff = torch.tensor(0, device=device)
        diff = y[1] - 0.6
        if diff > 0:
            diff = diff*5./4. + 0.5
        else:
            diff = diff*5./6. + 0.5
            
        #diff = y[1]
        
        #diff = (double) diff
        
        #diff = y[0][1]
        #print(diff)
        #print(y_target[i])
        
        #print(diff.device, y_target[i].device, y[1].device)
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
    #    print(params.grad)
        
    params.grad = params.grad/n_train
    optimizer.step()
    optimizer.zero_grad()
    

time_end = time.time()

d_erg = torch.tensor(np.random.rand(np.int(n_test/2), N)*2-1, device=device)*h_erg*h_bar*t_d*math.pi
d_local = torch.tensor(np.random.rand(np.int(n_test/2), N)*2-1, device=device)*h_loc*h_bar*t_d*math.pi
d = torch.cat((d_erg, d_local), 0)

y_target_test = torch.Tensor(np.array([1]*np.int(n_test/2)+[0]*np.int(n_test/2)))
y_target_test = y_target_test.double()
y_target_test = y_target_test.to(device)
y_list = [0 for _ in range(n_test)]

l_sum=0
target_list = [0 for _ in range(n_test)]

for i in range(n_test):
    #cir_params = torch.cat((params, -params[:,0].view(-1,1)),1)
    #n_params = params.view(-1)
    y = circuitDef(d[i], params)
    
    diff = y[1] - 0.6
    if diff > 0:
        diff = diff*5./4. + 0.5
    else:
        diff = diff*5./6. + 0.5
        
    #diff = y[1]
            
    l = loss(diff, y_target_test[i])

    l_sum = l_sum + l
    target_list[i] = y_target_test[i]
    y_list[i] = diff.data
    
#print(f'Testing: loss = {l_sum/n_test:.8f}')
#print("acc:", np.sum((np.round(y_list)==target_list))/n_test*100)
#print("prediction:  ", list(zip(y_list,target_list)))

print(' ')
print(' ')
print('number of qubits: ', n_size)
print('device: ', u_device)
print('pennylane time cost for 50 epochs, 40 trains per epoch: ', time_end-time_start, 's')
