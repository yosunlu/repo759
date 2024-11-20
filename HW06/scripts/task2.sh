#!/usr/bin/env bash
#SBATCH --job-name=task2
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="./output/task2.out"
#SBATCH --error="./output/task2.err"
#SBATCH --gres=gpu:1

rm -rf output
mkdir output

module load nvidia/cuda/11.8.0
module load gcc/9.4.0
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2
./task2
rm task2
