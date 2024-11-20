#!/usr/bin/env bash
#SBATCH --job-name=task2
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="./task2_c.out"
#SBATCH --error="./task2_c.err"
#SBATCH --gres=gpu:1


# Load required modules
module load nvidia/cuda/11.8.0
module load gcc/9.4.0

# Compile the CUDA program
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Define parameters
THREADS_PER_BLOCK1=1024
THREADS_PER_BLOCK2=64

R=128

# Loop through values of n
for ((i=10; i<=29; i++)); do
    N=$((2**i))
    echo "Running task1 with n=$N and threads_per_block=$THREADS_PER_BLOCK1"
    ./task2 $N $R $THREADS_PER_BLOCK1 >> task2_${THREADS_PER_BLOCK1}.txt

    echo "Running task1 with n=$N and threads_per_block=$THREADS_PER_BLOCK2"
    ./task2 $N $R $THREADS_PER_BLOCK2 >> task2_${THREADS_PER_BLOCK2}.txt
done

# Clean up
rm task2
