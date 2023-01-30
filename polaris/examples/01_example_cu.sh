#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A datascience
#PBS -o logs/
#PBS -e logs/

module load cudatoolkit-standalone/11.8.0

/path/to/ALCFBeginnersGuide/polaris/examples/example_cu
