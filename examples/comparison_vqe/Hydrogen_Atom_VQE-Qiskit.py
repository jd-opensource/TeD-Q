#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')
import torch
from qiskit import *
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient, PauliOp, Zero
from qiskit import quantum_info
from qiskit_machine_learning.neural_networks import OpflowQNN

import time

# Related package

import numpy as np
import matplotlib.pyplot as plt

# Hamiltonian related
from openfermion.chem import MolecularData
import openfermion
from openfermionpyscf import run_pyscf

# MISC
r_bohr = 0.529177210903

# Global variable
n_qubits = 4

# 
tolerance = 1e-6
min_pass = 5

# Set molecule parameters.
basis = 'sto-3g'
multiplicity = 1
n_points = 40
bond_length_interval = 3.0 / n_points
# Qiskit instance
qi = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))


# In[2]:


def Rot(circ, alpha, beta, theta, qubit):
    circ.rx(alpha, qubit)
    circ.ry(beta, qubit)
    circ.rz(theta, qubit)
# Ansatz
def ansatz(circ, params):
    for i in range(n_qubits):
        print(i)
        Rot(circ, params[i*3+0], params[i*3+1], params[i*3+2], i)
    for j in range(n_qubits-1, -1, -1):
        for k in range(j+1, n_qubits):
            circ.cx(j, k)


# In[3]:


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
    print('Hartree-Fock energy of {} Hartree.'.format(molecule.hf_energy))
    print('Hartree-Fock energy of {} Hartree.'.format(molecule.fci_energy))

    return jw_hamiltonian.terms, molecule.fci_energy


# In[4]:


# parameters
params_list = []
for idx in range(12):
    params_list.append(Parameter("param_"+str(idx)))


# In[5]:




def measurements(gatesPrefix):
    obs = list("IIII")
    for qubit, gatePrefix in list(gatesPrefix):
        obs[qubit]=gatePrefix

    return "".join(obs)
def initCircuit(distance):
    H, fci_energy = get_H2_hamiltonian(distance)
    def circuitDefTemplate(params, obsStr):
        ansatz(params)
        return  measurements(obsStr)
    
    qnn_circ = QuantumCircuit(n_qubits, 0)
    ansatz(qnn_circ, params_list)
    
    expval = AerPauliExpectation()
    
    # define gradient method
    gradient = Gradient()
    qc_sfn1 = StateFn(qnn_circ)
    weightList = []
    measList = []
    for i, idx in enumerate(H):
        weightList.append(H[idx].real)
        measList.append(measurements(idx))
        
    H1 = StateFn(PauliSumOp.from_list(list(zip(measList, weightList))))
    op1 = ~H1 @ qc_sfn1
    qnn = OpflowQNN(op1, [], qnn_circ.parameters, expval, gradient, qi)
    initial_weights = torch.ones(12, requires_grad=True)
    model = TorchConnector(qnn, initial_weights)

        
    return model, fci_energy


# In[7]:


import torch.nn as nn

distList = np.arange(0.5,1, 0.5)
energyList = np.array([])
fciEnergyList = np.array([])
print(distList)
params = torch.ones(12, requires_grad=True)
# params = torch.rand(12, requires_grad=True)
# params = torch.tensor([3.2203e+00, 5.0488e-02, 1.0000e+00, 3.1412e+00, 3.1412e+00, 1.0000e+00, 3.1407e+00, 1.0005e-03, 1.0000e+00, 4.9604e-04, 2.4310e-03, 1.0000e+00],requires_grad=True)

time_start = time.time()


timestamp = np.array([])
error = np.array([])

for distance in distList:
    model, fciE = initCircuit(distance)
#     params = torch.rand(12, requires_grad=True)
#     params = torch.tensor([3.2203e+00, 5.0488e-02, 1.0000e+00, 3.1412e+00, 3.1412e+00, 1.0000e+00, 3.1407e+00, 1.0005e-03, 1.0000e+00, 4.9604e-04, 2.4310e-03, 1.0000e+00],requires_grad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9)
    
#     print(weightList)
    exp = last_exp = 10000
    count = 0.
    for epoch in range(1000):
        l_sum = 0

        loss = nn.L1Loss()
        exp = model()
        
        
        if epoch%5==0:
            print(epoch, exp.item())
            timestamp = np.append(timestamp, time.time()-time_start)
            error = np.append(error, exp.item()-fciE)
        if np.abs(exp.item()-last_exp)<tolerance:
            count+=1
            if count>min_pass:
                timestamp = np.append(timestamp, time.time()-time_start)
                error = np.append(error, exp.item()-fciE)
                break
        else:
            count = 0
        last_exp = exp.item()
        l = loss(exp, torch.Tensor([-100.]))
        l.backward()        
        optimizer.step()
        optimizer.zero_grad()
    energyList = np.append(energyList, exp.item())
    fciEnergyList = np.append(fciEnergyList, fciE)
time_end = time.time()


# In[8]:


plt.plot(timestamp, error, 'r-', label="Qiskit")
plt.xlabel('Time(s)')
plt.grid()
plt.ylabel('Error')
plt.gca().set_yscale('log')
plt.legend()
print("time: ", time_end-time_start)


# In[9]:


import torch.nn as nn

distList = np.arange(0.5,3.5, 0.1)
energyList = np.array([])
fciEnergyList = np.array([])
print(distList)
params = torch.ones(12, requires_grad=True)
# params = torch.rand(12, requires_grad=True)
# params = torch.tensor([3.2203e+00, 5.0488e-02, 1.0000e+00, 3.1412e+00, 3.1412e+00, 1.0000e+00, 3.1407e+00, 1.0005e-03, 1.0000e+00, 4.9604e-04, 2.4310e-03, 1.0000e+00],requires_grad=True)

time_start = time.time()


for distance in distList:
    time_epoch_start = time.time()
    model, fciE = initCircuit(distance)
    params = torch.rand(12, requires_grad=True)
#     params = torch.tensor([3.2203e+00, 5.0488e-02, 1.0000e+00, 3.1412e+00, 3.1412e+00, 1.0000e+00, 3.1407e+00, 1.0005e-03, 1.0000e+00, 4.9604e-04, 2.4310e-03, 1.0000e+00],requires_grad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9)
    
#     print(weightList)
    exp = last_exp = 10000
    count = 0.
    for epoch in range(1000):
        l_sum = 0

        loss = nn.L1Loss()
        exp = model()
        
        
        if epoch%5==0:
            print(epoch, exp.item())
        if np.abs(exp.item()-last_exp)<tolerance:
            count+=1
            if count>min_pass:
                break
        else:
            count = 0
        if time.time()-time_epoch_start>10:
            break
        last_exp = exp.item()
        l = loss(exp, torch.Tensor([-100.]))
        l.backward()        
        optimizer.step()
        optimizer.zero_grad()
    energyList = np.append(energyList, exp.item())
    fciEnergyList = np.append(fciEnergyList, fciE)
time_end = time.time()


# In[10]:


plt.plot(distList, energyList, 'r-', label="VQE simulation")
plt.plot(distList, fciEnergyList, 'b--', label="FCI energy")
plt.xlabel('Distance(a0)')

plt.ylabel('Energy(Hartree)')
plt.legend()
print("time: ", time_end-time_start)


# In[11]:


np.savez('vqe_qiskit.npz', t=timestamp, e=error, d=distList, el=energyList, rel=fciEnergyList)


# ##### 

# In[ ]:




