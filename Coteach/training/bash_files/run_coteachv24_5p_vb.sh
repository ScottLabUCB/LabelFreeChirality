#!/bin/bash
# Job name:
#SBATCH --job-name=coteach_run_chiral_v24_5p_vb
#
# Account:
#SBATCH --account=fc_electron
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Specify one task:
#SBATCH --ntasks-per-node=1
#
# Number of processors for single task needed for use case (example):
#SBATCH --cpus-per-task=4
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:2
#
# Wall clock limit:
#SBATCH --time=5:00:00
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=cgroschner@berkeley.edu
#
## Command(s) to run (example):
module load cuda/9.0
module load pytorch/0.4.0-py36-cuda9.0
python main_chiral7.py --dataset chiral --noise_type chiral_flip --noise_rate 0.05 --forget_rate 0.05 --batch_size 128 --version_number 24b
