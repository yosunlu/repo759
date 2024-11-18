#!/usr/bin/env bash
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="./output/task1.out"
#SBATCH --error="./output/task1.err"
#SBATCH --gres=gpu:1

# Create the output folder if it doesn't exist
mkdir -p output

# Load required modules
module load nvidia/cuda/11.8.0
module load gcc/9.4.0

# Compile the CUDA program
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Define parameters
THREADS_PER_BLOCK1=64
THREADS_PER_BLOCK2=1024

# Loop through values of n
for ((i=5; i<=14; i++)); do
    N=$((2**i))
    echo "Running task1 with n=$N and threads_per_block=$THREADS_PER_BLOCK1"
    ./task1 $N $THREADS_PER_BLOCK1 >> output/task1_${THREADS_PER_BLOCK1}.txt

    echo "Running task1 with n=$N and threads_per_block=$THREADS_PER_BLOCK2"
    ./task1 $N $THREADS_PER_BLOCK2 >> output/task1_${THREADS_PER_BLOCK2}.txt
done

# Clean up
rm task1
