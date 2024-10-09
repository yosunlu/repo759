#!/usr/bin/env bash
#SBATCH --job-name=FirstSlurm
#SBATCH --partition=instruction
#SBATCH --cpus-per-task=2
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err

make clean
# Compile the program
make run_task3
./task3 >> FirstSlurm.out