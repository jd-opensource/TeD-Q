import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument(
        "--world_size",
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
        "--num_gpus",
        type=int,
        default=0,
        help="""Number of GPUs to use for training, Currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="5678",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    assert args.world_size is not None, "must provide world_size argument."
    assert args.rank is not None, "must provide rank argument."
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    world_size = args.world_size
    rank = args.rank
    num_gpus = args.num_gpus
    master_addr = args.master_addr
    master_port = args.master_port

    from jdtensorpath.distributed import run_distributed
    run = run_distributed(world_size, rank, num_gpus, master_addr, master_port)
    run.shutdown()
