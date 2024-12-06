#!/usr/bin/env bash
#SBATCH --job-name=task2_b
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2_b
#SBATCH --time=0-00:30:00
#SBATCH --output="./task2.out"
#SBATCH --error="./task2.err"
#SBATCH --gres=gpu:1

# Load necessary modules
module load nvidia/cuda/11.8.0
module load gcc/9.4.0

# Compile the program
nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Run the program for n = 2^10 to 2^30 with threads_per_block=256
for ((i=10; i<=30; i++)); do
    n=$((2**i))
    echo "Running task2 for n = $n with threads_per_block=256" >> task2.out
    ./task2 $n 256 >> task2.out
    echo "Finished task2 for n = $n with threads_per_block=256" >> task2.out
done

# Run the program for n = 2^10 to 2^30 with threads_per_block=1024
for ((i=10; i<=30; i++)); do
    n=$((2**i))
    echo "Running task2 for n = $n with threads_per_block=1024" >> task2.out
    ./task2 $n 1024 >> task2.out
    echo "Finished task2 for n = $n with threads_per_block=1024" >> task2.out
done

# Clean up
rm task2
