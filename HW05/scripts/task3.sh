#!/usr/bin/env bash
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="./out/task3.out"
#SBATCH --error="./err/task3.err"
#SBATCH --gres=gpu:1
#SBATCH --mem=16G


module load nvidia/cuda/11.8.0
module load gcc/9.4.0
nvcc ./task3.cu ./vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

for ((i=10; i<30; i++)); do
	n=$((2**i))
	./task3 $n >> ./out/task3.out
done
rm task3
