# ECE 759 - High Performance Computing Assignments Overview

This repository contains the implementations for the assignments from the **ECE 759 - High Performance Computing for Engineering Applications** course. Each assignment focuses on optimizing computational tasks, exploring parallelization techniques, and analyzing performance for large-scale datasets.

## Assignments Overview

### Assignment 1: Introduction to High Performance Computing
- **Tasks**: 
  - Getting familiar with Linux, Slurm, and Euler computing environment.
  - Writing bash scripts for job submissions and simple command-line tasks.
  - Implementing a C++ program to print numbers in ascending and descending order.
- **Focus**: Learn foundational HPC tools and workflows.

### Assignment 2: Convolutions and Matrix Multiplication (Row/Column Major Analysis)
- **Tasks**:
  - Implement inclusive scan (prefix sum) for large arrays.
  - 2D Convolution operations with specific boundary padding rules.
  - Matrix multiplication with various loop orders (row/column major) and performance comparisons.
- **Focus**: Analyze cache efficiency and scalability of computational algorithms.

### Assignment 3: Parallelized Algorithms with OpenMP
- **Tasks**:
  - Parallelized matrix multiplication using OpenMP.
  - 2D Convolutions on large matrices with thread optimizations.
  - Merge sort algorithm parallelized with OpenMP tasks, analyzing thread count and task size thresholds.
- **Focus**: Gain experience with OpenMP and analyze the trade-offs in parallel performance.

### Assignment 4: N-Body Simulation with OpenMP
- **Tasks**:
  - Implement gravitational N-body simulation in Python and C++.
  - Parallelize the C++ implementation with OpenMP.
  - Experiment with different OpenMP scheduling policies.
- **Focus**: Explore particle simulation and analyze parallel scheduling strategies.

### Assignment 5: CUDA Programming Basics
- **Tasks**:
  - Compute factorials using CUDA kernels.
  - Perform array computations using thread and block indices.
  - Implement vector scaling using CUDA.
- **Focus**: Get started with GPU programming using CUDA.

### Assignment 6: Advanced CUDA Techniques
- **Tasks**:
  - Tiled matrix multiplication using CUDA shared memory.
  - 1D convolution using stencil operations with shared memory optimizations.
- **Focus**: Optimize memory access patterns in CUDA programs.

### Assignment 7: Matrix Multiplication and Reduction with CUDA
- **Tasks**:
  - Tiled matrix multiplication for large matrices.
  - Parallel reduction for summing large arrays with optimized kernels.
- **Focus**: Advanced CUDA programming and performance benchmarking.

---

This repository showcases the progression from foundational HPC techniques to advanced CUDA programming, emphasizing optimization, scalability, and parallelization strategies.
