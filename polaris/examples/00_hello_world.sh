#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -q R2011010
#PBS -l filesystems=home
#PBS -A alcf_training
#PBS -o logs/
#PBS -e logs/

module use /soft/modulefiles
module load cudatoolkit-standalone/12.4.0

GPUS_PER_NODE=4

mpiexec -n $GPUS_PER_NODE -ppn $GPUS_PER_NODE echo Hello World
