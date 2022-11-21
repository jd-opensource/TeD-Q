#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../..')
import tedq as qai

import pennylane as qml

# Related package
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

import gpytorch


# Hamiltonian related
from openfermion.chem import MolecularData
import openfermion
from openfermionpyscf import run_pyscf

# MISC
r_bohr = 0.529177210903

# Global variable
n_qubits = 4
n_layers = 3
n_search = 500
n_experts = 5

# 
tolerance = 1e-6
min_pass = 5

# Set molecule parameters.
basis = 'sto-3g'
multiplicity = 1
n_points = 40
bond_length_interval = 3.0 / n_points

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[2]:


import itertools


valid_Rs =  [qai.RY, qai.RZ]
valid_CNOTs = ([0, 1], [1, 2], [2, 3])

Rs_space = list(itertools.product(valid_Rs, valid_Rs, valid_Rs, valid_Rs))
CNOTs_space = [[y for y in CNOTs if y is not None] for CNOTs in list(itertools.product(*([x, None] for x in valid_CNOTs)))]
NAS_search_space = list(itertools.product(Rs_space, CNOTs_space))


# In[3]:


len(NAS_search_space)


# In[4]:


NAS_search_space[3][1]


# In[5]:


def Rot(alpha, beta, theta, qubit):
    qml.RX(alpha, wires=qubit)
    qml.RY(beta, wires=qubit)
    qml.RZ(theta, wires=qubit)
# Ansatz
def ansatz(params):
    for i in range(n_qubits):
        Rot(params[i][0], params[i][1], params[i][2], i)
    for j in range(n_qubits-1, -1, -1):
        for k in range(j+1, n_qubits):
            qml.CNOT(wires=[j, k])


# In[6]:


# Hamiltonian


def get_H2_hamiltonian(distance):
    bond_length=distance*r_bohr
    geometry = [('H', (0., 0., -bond_length/2)), ('H', (0., 0., bond_length/2))]
    molecule = MolecularData(
        geometry, basis, multiplicity,charge=0,
        description=str(round(bond_length, 2)))
    molecule = run_pyscf(molecule,run_scf=1,run_fci=1)
    terms_molecular_hamiltonian = molecule.get_molecular_hamiltonian(occupied_indices=[], active_indices=[0,1])
    fermionic_hamiltonian = openfermion.transforms.get_fermion_operator(terms_molecular_hamiltonian)
    jw_hamiltonian = openfermion.transforms.jordan_wigner(fermionic_hamiltonian)
#     print(jw_hamiltonian.terms)
#     print('Hartree-Fock energy of {} Hartree.'.format(molecule.hf_energy))
#     print('Hartree-Fock energy of {} Hartree.'.format(molecule.fci_energy))

    return jw_hamiltonian.terms, molecule.fci_energy


# In[7]:


AIdList = []

# Ansatz

def ansatz(params, AIdList):
    for i, Aid in enumerate(AIdList):
        for j in range(n_qubits):
            NAS_search_space[Aid][0][j](params[i][j], qubits=[j])
        for cnot_pair in NAS_search_space[Aid][1]:
            qai.CNOT(qubits=list(cnot_pair))


# In[8]:



def getGateFromName(name):
    if name=="I":
        return qai.I
    if name=="X":
        return qai.PauliX
    if name=="Y":
        return qai.PauliY
    if name=="Z":
        return qai.PauliZ
def measurements(gatesPrefix):
    gateList = []
    for qubit, gatePrefix in list(gatesPrefix):
        gate = getGateFromName(gatePrefix)
        gateList.append(gate(qubits=[qubit]))
    return gateList

def initCircuit(distance, selectedAIdList):
    AIdList = selectedAIdList
    H, fci_energy = get_H2_hamiltonian(distance)
    for idx in H:
        def circuitDef(params):
            ansatz(params, AIdList)
            qai.measurement.expval(measurements(idx))
        circList.append(qai.Circuit(circuitDef, n_qubits, torch.zeros(n_layers,n_qubits)))
        compiledCircList.append(circList[-1].compilecircuit('pytorch', print_output=False))
        weightList.append(H[idx])
    return fci_energy


# In[9]:


def expert_evaluator(model, subnet, n_experts, cost_fn):

    target_expert = 0
    target_loss = None
    for i in range(n_experts):
        model.params = model.get_params(subnet, i)
        temp_loss = cost_fn(model.params)
        if target_loss is None or temp_loss < target_loss:
            target_loss = temp_loss
            target_expert = i
    return target_expert


# In[10]:


class CircuitSearchModel():
    def __init__(self, n_qubits=4, n_layers=3, n_experts=5):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_experts = n_experts
        '''init params'''
        # randomly initialize parameters from a normal distribution
        self.params_space = np.random.uniform(0, np.pi * 2, (n_experts, n_layers, len(Rs_space), n_qubits))
        self.params = torch.ones((3,4), requires_grad=True)
    
    def get_params(self, subnet, expert_idx):
        self.subnet = subnet
        self.expert_idx = expert_idx
        params = []
        # print("CNOT", NAS_search_space)
        # print("get", subnet, expert_idx)
        for j in range(self.n_layers):
            r_idx = subnet[j] // len(CNOTs_space)
            params.append(self.params_space[expert_idx, j, r_idx:r_idx+1])
#         print("get param", params, np.concatenate(params, axis=0))
        self.params=torch.tensor(params, requires_grad=True)
        return np.concatenate(params, axis=0)

    def set_params(self):
        self.params = self.params.cpu().detach().numpy()
        for j in range(self.n_layers):
            r_idx = self.subnet[j] // len(CNOTs_space)
            # print("set", j, self.subnet[j], r_idx, len(CNOTs_space))
            # print("set", self.params_space)
            self.params_space[self.expert_idx, j, r_idx:r_idx+1] = self.params[j, :]

    def __call__(self, params, wires):
        circuit_search(params, wires=wires,
                       n_qubits=self.n_qubits,
                       n_layers=self.n_layers,
                       arch=self.subnet)


# In[11]:


import torch.nn as nn

distList = np.arange(0.5,1.0, 0.5)
energyList = np.array([])
timeList = np.array([])
fciEnergyList = np.array([])
modelParam = CircuitSearchModel()

timestamp = np.array([])
error = np.array([])

distance = 1.2


# In[12]:


step_size = 0.2


time_start=time.time()    


for i_iter in range(n_search):
    circList = []
    compiledCircList = []
    weightList = []
    selectedAIdList = np.random.randint(0, len(NAS_search_space), (n_layers,)).tolist()
    
    fciE = initCircuit(distance, selectedAIdList)
    
#     find optimal expert
    expert_idx = np.random.randint(n_experts)
    if i_iter>100:
        for i in range(n_experts):
            # get params
            min_loss = 100000
            min_expert_id = 0
            params = modelParam.get_params(selectedAIdList, i)
#             print("params",params)
            # calculate
            exp = 0
            for idx, compiledCirc in enumerate(compiledCircList):   
#                 print("compile idx ", idx)
                exp += compiledCirc(modelParam.params).real*weightList[idx].real
            loss = exp
            if loss<min_loss:
                min_loss = loss
                min_expert_id = i
        expert_idx = min_expert_id
    # get params
    modelParam.get_params(selectedAIdList, expert_idx)
#     params = torch.zeros((3,4))
    vNgdOpt = gpytorch.optim.NGD([modelParam.params], num_data=1, lr=0.1)
    
    # for each set
    l_sum = 0
    loss = nn.L1Loss()
    exp = 0
    for idx, compiledCirc in enumerate(compiledCircList):   
        exp += compiledCirc(modelParam.params).real*weightList[idx].real


    l = loss(exp, torch.Tensor([-100.]))
    l.backward()        
#     optimizer.step()
#     optimizer.zero_grad()
#     params = opt.step(cost, params)
#     print("old", modelParam.params)
    vNgdOpt.step()
#     print("new", modelParam.params)
#     print("grad", modelParam.params.grad)
    vNgdOpt.zero_grad()

    # set params
    modelParam.set_params()
        
        
#     print(params)
    print(i_iter, selectedAIdList, fciE, exp.item())
    energyList = np.append(energyList, exp.item())
    timeList = np.append(timeList, time.time()-time_start)
    fciEnergyList = np.append(fciEnergyList, fciE)
time_end=time.time()


# In[ ]:


plt.plot(timeList, range(len(timeList)))
plt.xlabel("time(s)")
plt.ylabel("epoch size")


# In[ ]:


result = {}
for i_iter in range(n_search):
    circList = []
    compiledCircList = []
    weightList = []
    selectedAIdList = np.random.randint(0, len(NAS_search_space), (n_layers,)).tolist()
    
    fciE = initCircuit(distance, selectedAIdList)
    
#     find optimal expert
    for i in range(n_experts):
        # get params
        params = []
        min_loss = 100000
        min_expert_id = 0
        params = modelParam.get_params(selectedAIdList, i)

        # calculate
        exp = 0
        for idx, compiledCirc in enumerate(compiledCircList):   
            exp += compiledCirc(modelParam.params).real*weightList[idx].real
        loss = exp
        if loss<min_loss:
            min_loss = loss
            min_expert_id = i
    expert_idx = min_expert_id
    # get params
    params = []
    params = modelParam.get_params(selectedAIdList, expert_idx) 
    
    # for each set
    exp = 0
    for idx, compiledCirc in enumerate(compiledCircList):   
        exp += compiledCirc(modelParam.params).real*weightList[idx].real
        pass

        
        
#     print(params)
    print(i_iter, selectedAIdList, fciE, exp.item())
    
    result['-'.join([str(x) for x in selectedAIdList])] = min_loss
    
time_end=time.time()


# In[ ]:


print(time.time()-time_start)


# In[ ]:


# result = {}
# print(n_search)
# for i_iter in range(n_search):
#     selectedAIdList = np.random.randint(0, len(NAS_search_space), (n_layers,)).tolist()
#     circList = []
#     compiledCircList = []
#     weightList = []
#     fciE = initCircuit(distance, selectedAIdList)
#     for i in range(n_experts):
# #         print(i)
#         # get params
#         params = []
#         min_loss = 100000
#         min_expert_id = 0
#         for j in range(n_layers):
#             r_idx = selectedAIdList[j] // len(CNOTs_space)
#             params.append(params_space[i, j, r_idx:r_idx+1])
#         params = torch.tensor(np.concatenate(params, axis=0), requires_grad=False)  

#         # calculate
#         exp = 0
#         for idx, compiledCirc in enumerate(compiledCircList):   
#             exp += compiledCirc(params).real*weightList[idx].real
# #     print(exp)
#         loss = exp
#         if loss<min_loss:
#             min_loss = loss
#             min_expert_id = i
#     print(i_iter, selectedAIdList, min_loss)
#     result['-'.join([str(x) for x in selectedAIdList])] = min_loss
    


# In[ ]:


sorted_result = list(result.items())
sorted_result.sort(key=lambda x: x[1], reverse=True)
for idx, data in enumerate(sorted_result):
    print(idx, data[0],data[1].item())


# In[ ]:


np.savez('qas_tedq.npz', t=timeList, e=energyList, r=sorted_result)
