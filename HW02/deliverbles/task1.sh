#!/usr/bin/env bash
#SBATCH --job-name=FirstSlurm
#SBATCH --partition=instruction
#SBATCH --cpus-per-task=2
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err

make clean
# Compile the program
make run_task1

# Loop over each power of 2 from 2^10 to 2^30
for ((i=10; i<=30; i++)); do
    n=$((2**i))
    echo "Running task1 with n = $n" >> FirstSlurm.out
    ./task1 "$n" >> FirstSlurm.out
done