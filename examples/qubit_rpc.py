import tedq as qai

n_qubits = 10
q_depth = 1

# Define quantum circuit
def circuitDef(params):
    qai.templates.HardwareEfficient(n_qubits, q_depth, params)
    return qai.expval(qai.PauliZ(qubits=[0]))
    #return [qai.expval(qai.PauliZ(qubits=[idx])) for idx in range(n_qubits)]


parameter_shapes = [((q_depth+1)*2,n_qubits)]

# Quantum circuit construction
circuit = qai.Circuit(circuitDef, n_qubits, parameter_shapes = parameter_shapes)

my_compilecircuit = circuit.compilecircuit(backend="pytorch")

# from jdtensorpath import JDOptTN as jdopttn
# slicing_opts = {'target_size':2**28, 'repeats':1, 'target_num_slices':2, 'contract_parallel':False} #'distributed_CPU'
# hyper_opt = {'methods':['kahypar'], 'max_time':120, 'max_repeats':1, 'search_parallel':False, 'slicing_opts':slicing_opts}
# my_compilecircuit = circuit.compilecircuit(backend="pytorch", use_jdopttn=jdopttn, hyper_opt = hyper_opt, tn_simplify = False)

def cost(*params):
    results = my_compilecircuit(*params)
    result = sum(results)
    return result

import torch
from torch import optim
from jdtensorpath.distributed import run_distributed

run = run_distributed(2, 0)

parameters = torch.rand(((q_depth+1)*2,n_qubits), requires_grad= True)
#run.set_circuit(cost, [parameters], optim.Adam)
run.set_circuit(my_compilecircuit)

from torch import optim
optimizer = optim.Adam([parameters], lr=0.1)

for i in range(100):
    optimizer.zero_grad()
    result = run(parameters)
    optimizer.step()
    print(result)
    
run.shutdown()