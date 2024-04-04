#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A <project-name>
#PBS -o <your_log_dir>
#PBS -e <your_log_dir>

module load gcc/11.2.0

GPUS_PER_NODE=4

mpiexec -n $GPUS_PER_NODE -ppn $GPUS_PER_NODE echo Hello World
