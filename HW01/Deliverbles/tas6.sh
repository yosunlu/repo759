#!/usr/bin/env bash
#SBATCH --job-name=FirstSlurm
#SBATCH --partition=instruction
#SBATCH --cpus-per-task=2
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err

g++ task6.cpp -Wall -O3 -std=c++17 -o task6
./task6 6
