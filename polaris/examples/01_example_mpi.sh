#!/bin/bash
#PBS -l select=2
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A <project-name>
#PBS -o logs/
#PBS -e logs/

# Count number of nodes assigned
NNODES=`wc -l < $PBS_NODEFILE`
# set 1 MPI rank per GPU
NRANKS_PER_NODE=4
# calculate total ranks
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} $HOME/ALCFBeginnersGuide/polaris/examples/01_example_mpi
