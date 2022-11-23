import tedq as qai

n_qubits = 10
q_depth = 1

# Define quantum circuit
def circuitDef(features, params):
    qai.RX(features[0], qubits=[0])
    qai.RY(features[1], qubits=[1])
    qai.templates.HardwareEfficient(n_qubits, q_depth, params)
    return qai.expval(qai.PauliZ(qubits=[0]))
    #return [qai.expval(qai.PauliZ(qubits=[idx])) for idx in range(n_qubits)]


parameter_shapes = [(2,), ((q_depth+1)*2,n_qubits)]

# Quantum circuit construction
circuit = qai.Circuit(circuitDef, n_qubits, parameter_shapes = parameter_shapes)

my_compilecircuit = circuit.compilecircuit(backend="pytorch")

# from jdtensorpath import JDOptTN as jdopttn
# slicing_opts = {'target_size':2**28, 'repeats':1, 'target_num_slices':2, 'contract_parallel':False} #'distributed_CPU'
# hyper_opt = {'methods':['kahypar'], 'max_time':120, 'max_repeats':1, 'search_parallel':False, 'slicing_opts':slicing_opts}
# my_compilecircuit = circuit.compilecircuit(backend="pytorch", use_jdopttn=jdopttn, hyper_opt = hyper_opt, tn_simplify = False)

# first parameter must be quantum_results: a list object, the results of quantum circuit
# 
def cost(quantum_results, *cost_parameters):
    result = sum(quantum_results)/len(quantum_results)*cost_parameters[0] + cost_parameters[1]
    #print(len(quantum_results))
    return result



import torch
from torch import optim
from jdtensorpath.distributed import run_distributed

# world_size, rank, num_gpus
run = run_distributed(2, 0, 4)

parameters = torch.rand(((q_depth+1)*2,n_qubits), requires_grad= True)
#run.set_circuit(cost, [parameters], optim.Adam)
#cost_func = cost(my_compilecircuit)
run.set_circuit(my_compilecircuit)
run.set_cost_func(cost)

from torch import optim
optimizer = optim.Adam([parameters], lr=0.1)

features = torch.rand((2,), requires_grad= False)

for i in range(100):
    optimizer.zero_grad()
    # (tuple(quantum_parameters), tuple(cost_parameters))
    result = run(qauntum_parameters=(features, parameters), cost_parameters=(1., 2.))
    optimizer.step()
    print(result)
    
run.shutdown()