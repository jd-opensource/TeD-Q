import argparse

import tedq as qai




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=None,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument(
        "--gpus_per_cpu",
        type=int,
        default=0,
        help="""Number of GPUs to use for training, Currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
    parser.add_argument(
        "--cpus_per_node",
        type=int,
        default=1,
        help="""Number of GPUs to use for training, Currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",#172.17.224.178
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="8119",#8119
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    assert args.num_nodes is not None, "must provide num_nodes argument."
    assert args.rank is not None, "must provide rank argument."

    num_nodes = args.num_nodes
    rank = args.rank
    gpus_per_cpu = args.gpus_per_cpu
    cpus_per_node = args.cpus_per_node
    master_addr = args.master_addr
    master_port = args.master_port



n_qubits = 10
q_depth = 1

# Define quantum circuit
def circuitDef(features, params):
    qai.RX(features[0], qubits=[1])
    qai.RY(features[1], qubits=[1])
    qai.templates.HardwareEfficient(n_qubits, q_depth, params)
    return qai.expval(qai.PauliZ(qubits=[0]))
    #return [qai.expval(qai.PauliZ(qubits=[idx])) for idx in range(n_qubits)]


parameter_shapes = [(2,), ((q_depth+1)*2,n_qubits)]

# Quantum circuit construction
circuit = qai.Circuit(circuitDef, n_qubits, parameter_shapes = parameter_shapes)

# my_compilecircuit = circuit.compilecircuit(backend="pytorch")

from jdtensorpath import JDOptTN as jdopttn
slicing_opts = {'target_size':2**28, 'repeats':1, 'target_num_slices':2, 'contract_parallel':'distributed_CPU'} #'distributed_CPU'
hyper_opt = {'methods':['kahypar'], 'max_time':120, 'max_repeats':1, 'search_parallel':False, 'slicing_opts':slicing_opts}
my_compilecircuit = circuit.compilecircuit(backend="pytorch", use_jdopttn=jdopttn, hyper_opt = hyper_opt, tn_simplify = False)

# first parameter must be quantum_results: a list object, the results of quantum circuit
# 
def cost(quantum_results, *cost_parameters):
    result = sum(quantum_results)/len(quantum_results)*cost_parameters[0] + cost_parameters[1]
    #print(len(quantum_results))
    return result

def cost_1(features, params):
    result = 1*my_compilecircuit(features, params)  + 2
    return result


import torch
from torch import optim
# from jdtensorpath.distributed import run_distributed_circuit_parallel

# num_nodes, rank=0, gpus_per_cpu=0, cpus_per_node=1, master_addr='localhost', master_port='8119'
# run = run_distributed_circuit_parallel(num_nodes, rank, gpus_per_cpu, cpus_per_node)

parameters = torch.rand(((q_depth+1)*2,n_qubits), requires_grad= True)
# run.set_circuit(my_compilecircuit)
# run.set_cost_func(cost)



from jdtensorpath.distributed import run_distributed_slicing_parallel

# num_nodes, rank=0, gpus_per_cpu=0, cpus_per_node=1, master_addr='localhost', master_port='8119'
run = run_distributed_slicing_parallel(num_nodes, rank, gpus_per_cpu, cpus_per_node, master_addr, master_port)
run.set_cost_func(cost_1)

from torch import optim
optimizer = optim.Adam([parameters], lr=0.1)

features = torch.rand((2,), requires_grad= False)

for i in range(100):
    optimizer.zero_grad()
    # (tuple(quantum_parameters), tuple(cost_parameters))
    # result = run(quantum_parameters=[(features, parameters), (features, parameters)], cost_parameters=(1., 2.))
    result = run(features, parameters)
    optimizer.step()
    print(result)
    
run.shutdown()