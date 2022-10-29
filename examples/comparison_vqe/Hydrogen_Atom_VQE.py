#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')
import tedq as qai

# Related package
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[2]:


def Rot(alpha, beta, theta, qubit):
    qai.RX(alpha, qubits=[qubit])
    qai.RY(beta, qubits=[qubit])
    qai.RZ(theta, qubits=[qubit])
# Ansatz
def ansatz(params):
    for i in range(n_qubits):
        Rot(params[i][0], params[i][1], params[i][2], i)
    for j in range(n_qubits-1, -1, -1):
        for k in range(j+1, n_qubits):
            qai.CNOT(qubits=(j, k))


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
    for qubit, gatePrefix in list(gatesPrefix):
        gate = getGateFromName(gatePrefix)
        gate(qubits=[qubit])

def initCircuit(distance):
    H, fci_energy = get_H2_hamiltonian(distance)
    for idx in H:
        def circuitDef(params):
            ansatz(params)
            measurements(idx)
            qai.measurement.state()
        circList.append(qai.Circuit(circuitDef, n_qubits, torch.zeros(n_qubits,3)))
        compiledCircList.append(circList[-1].compilecircuit('pytorch'))
        weightList.append(H[idx])
    return fci_energy


# In[5]:


import torch.nn as nn

distList = np.arange(0.5,1.0, 0.5)
energyList = np.array([])
fciEnergyList = np.array([])
print(distList)
params = torch.ones(12, requires_grad=True)
# params = torch.rand(12, requires_grad=True)

time_start=time.time()    

timestamp = np.array([])
error = np.array([])

for distance in distList:
    circList = []
    compiledCircList = []
    weightList = []
    fciE = initCircuit(distance)
#     params = torch.ones(12, requires_grad=True)
#     params = torch.rand(12, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=0.9)

    exp = last_exp = 10000
    count = 0
    for epoch in range(200):
        l_sum = 0

        loss = nn.L1Loss()
        x = torch.reshape(params, (4,3))
        psi_star = torch.conj(compiledCircList[0](x))
        exp = 0
        for idx, compiledCirc in enumerate(compiledCircList):   
            H_psi = compiledCirc(x)
            exp += torch.tensordot(psi_star[0],H_psi[0], dims=4).real*weightList[idx].real
            pass
        
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
    print(params)
    energyList = np.append(energyList, exp.item())
    fciEnergyList = np.append(fciEnergyList, fciE)
time_end=time.time()


# In[6]:


plt.plot(timestamp, error, 'r-', label="TeD-Q")
plt.xlabel('Time(s)')
plt.grid()
plt.ylabel('Error')
plt.gca().set_yscale('log')
plt.legend()
print("time: ", time_end-time_start)


# In[7]:


distList = np.arange(0.5,3.5, 0.1)
energyList = np.array([])
fciEnergyList = np.array([])
print(distList)
params = torch.ones(12, requires_grad=True)
# params = torch.rand(12, requires_grad=True)

time_start=time.time()    


for distance in distList:
    time_epoch_start = time.time()
    circList = []
    compiledCircList = []
    weightList = []
    fciE = initCircuit(distance)
#     params = torch.ones(12, requires_grad=True)
#     params = torch.rand(12, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=0.9)

    exp = last_exp = 10000
    count = 0
    for epoch in range(200):
        l_sum = 0

        loss = nn.L1Loss()
        x = torch.reshape(params, (4,3))
        psi_star = torch.conj(compiledCircList[0](x))
        exp = 0
        for idx, compiledCirc in enumerate(compiledCircList):   
            H_psi = compiledCirc(x)
            exp += torch.tensordot(psi_star[0],H_psi[0], dims=4).real*weightList[idx].real
            pass
        
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
    print(params)
    energyList = np.append(energyList, exp.item())
    fciEnergyList = np.append(fciEnergyList, fciE)
time_end=time.time()


# In[8]:


plt.plot(distList, energyList, 'r-', label="VQE simulation")
plt.plot(distList, fciEnergyList, 'b--', label="FCI energy")
plt.xlabel('Distance(a0)')

plt.ylabel('Energy(Hartree)')
plt.legend()
print("time: ", time_end-time_start)


# In[9]:


np.savez('vqe_tedq.npz', t=timestamp, e=error, d=distList, el=energyList, rel=fciEnergyList)


# In[ ]:




