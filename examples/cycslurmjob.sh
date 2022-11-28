#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=p40
#SBATCH -J myFirstJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1

cd /raid/slurm-for-quantum/home/qc01/cyc/TeD-Q/tedq/distributed_worker/
rank=$(($SLURM_PROCID+1))
python rpc_workers.py --num_nodes 1 --rank $rank --gpus_per_cpu 0 --cpus_per_node 2
#hostname
#pwd
#$SLURM_PROCID
#$(SLURM_PROCID)
#rank=$(($SLURM_PROCID+1))
#echo $rank 
#python ../../examples/fuck.py $rank
