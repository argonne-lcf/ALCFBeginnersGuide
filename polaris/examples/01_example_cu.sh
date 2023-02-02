#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A <project-name>
#PBS -o logs/
#PBS -e logs/

module load cudatoolkit-standalone/11.8.0

$HOME/ALCFBeginnersGuide/polaris/examples/01_example_cu
