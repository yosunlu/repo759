# Assignment 2 - Convolutions, and Matrix Multiplication (Row/Column Major Analysis)
## Overview
This repository contains the code and implementation for Assignment 2 of the ECE 759 - High-Performance Computing course. The assignment focuses on implementing key computational algorithms and analyzing their performance on large-scale datasets. The key tasks include:

- **Scan Function**: Implemented an inclusive scan (prefix sum) algorithm and evaluated its performance for increasing input sizes from 2^10 to 2^30. Plotted the execution time as a function of input size.
- **Convolutions**: Implemented a 2D convolution operation on an n×n matrix with an m×m mask, using specific boundary padding rules (zeros for corners, ones for edges). Measured the time taken for convolution on different input sizes.
- **Matrix Multiplication**: Implemented four different matrix multiplication algorithms (mmul1, mmul2, mmul3, mmul4) based on different loop orders (row-major and column-major). Compared their performance using large matrices.

## Analysis
*Documentation and methods for timing and ramdonization can be found in the root folder* 
### Task1
![截圖 2024-10-09 上午1 49 15](https://github.com/user-attachments/assets/a3b4947b-48ff-4c74-b8ae-c127bbc2d0ed)
The plot shows the scaling analysis of the scan function in Task 1, with execution time increasing logarithmically as the input array size grows from 2^9 to 2^{30}. The straight-line trend in the log-log plot indicates a linear relationship between array size and execution time, suggesting that the scan function scales efficiently as the input size increases. Performance is consistent, with minimal overhead for small arrays and predictable increases for larger ones, demonstrating that the implementation effectively handles larger datasets without significant bottlenecks.

### Task3
With the performance order 2 < 1 ≈ 4 < 3 :
- mmul2 (i, k, j) is the fastest because the middle loop ( k ) allows accessing B in a row-wise manner, which improves cache locality, minimizing cache misses and making the computation efficient.
- mmul1 (i, j, k) and mmul4 perform similarly. In both, accessing B column-wise (j) leads to inefficient, non-contiguous memory access, resulting in more cache misses. The overhead of std::vector in mmul4 roughly cancels out the impact of the memory access pattern, leading to performance similar to mmul1.
- mmul3 (j, k, i) is the slowest because accessing C column-wise ( j ) for every k iteration causes inefficient memory usage. This poor cache locality results in more cache misses, thus significantly degrading performance compared to the other implementations.

## How to run
*These tasks can be run on non-euler machines.*
### Non Euler machines
After cloning the repo, cd to the deliverbles directory

`cd ../path_to_deliverbles`

Run the makefile (using task 1 for example)  

`make run_task1`
### Euler machines
Run the shell scripts directly    

`./task1.sh`
