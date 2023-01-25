#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A datascience

module load cudatoolkit-standalone gcc

./example_cu
