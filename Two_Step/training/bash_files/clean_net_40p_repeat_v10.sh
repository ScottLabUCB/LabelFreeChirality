#!/bin/bash
# Job name:
#SBATCH --job-name=clean_net_40p_v10
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
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=14:00:00
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=cgroschner@berkeley.edu
#
## Command(s) to run (example):
module load ml/tensorflow/1.12.0-py36
python clean_net_main.py --noise_rate 0.4 --lr 0.001 --n_epoch_net1 10 --n_epoch_net2 100 --repeats 1 --version_number 10
