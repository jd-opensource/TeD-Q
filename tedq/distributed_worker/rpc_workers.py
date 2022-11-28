import argparse
import os

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

    num_nodes = int(args.num_nodes)
    rank=int(os.environ['SLURM_PROCID']) + 1 # rank = args.rank
    gpus_per_cpu = args.gpus_per_cpu
    cpus_per_node = args.cpus_per_node
    master_addr = args.master_addr
    master_port = args.master_port


    from jdtensorpath.distributed import run_distributed

    # num_nodes, rank=0, gpus_per_cpu=0, cpus_per_node=1, master_addr='localhost', master_port='8119'
    run = run_distributed(num_nodes, rank, gpus_per_cpu, cpus_per_node, master_addr, master_port)
    run.shutdown()