#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=p40
#SBATCH -J myFirstJob
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

cd /raid/slurm-for-quantum/home/qc01/cyc/TeD-Q/tedq/distributed_worker/
rank=$(($SLURM_PROCID+1))
srun python rpc_workers.py --num_nodes 2 --rank $rank --gpus_per_cpu 4 --cpus_per_node 1 --master_addr 172.17.224.177
#hostname
#pwd
#$SLURM_PROCID
#$(SLURM_PROCID)
#rank=$(($SLURM_PROCID+1))
#echo $rank 
#python ../../examples/fuck.py $rank
