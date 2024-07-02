#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A <project-name>
#PBS -o <your_log_dir>
#PBS -e <your_log_dir>



$HOME/ALCFBeginnersGuide/polaris/examples/01_example_cu
