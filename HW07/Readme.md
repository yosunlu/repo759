# Tiled Matrix Multiplication and Parallel Reduction
This project contains CUDA implementations for high-performance computing tasks as part of ECE 759. It includes:
- Tiled Matrix Multiplication (matmul.cu): Implements shared memory optimization for matrix multiplication, supporting arbitrary matrix dimensions up to  2^14
- Parallel Reduction (reduce.cu): Efficiently sums large arrays using optimized reduction techniques.
Both implementations are benchmarked on the Euler cluster using Slurm, with performance analysis provided in HW07_deliverables.  
Refer to makefile to compile the files using on euler.
