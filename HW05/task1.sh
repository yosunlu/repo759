#!/usr/bin/env bash
#SBATCH --job-name=FirstSlurm
#SBATCH --partition=instruction
#SBATCH --gpus-per-task=1 
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err

module load nvidia/cuda/11.8.0
module load gcc/9.4.0
nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
compute-sanitizer --target-processes=all ./task1
