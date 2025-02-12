#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A Catalyst
#PBS -o logs/
#PBS -e logs/

GPUS_PER_NODE=6

mpiexec -n $GPUS_PER_NODE -ppn $GPUS_PER_NODE echo Hello World
