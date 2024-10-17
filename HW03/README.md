# Assignment 3 - Matrix Multiplication, Convolutions, and Merge Sort (Parallelized with OpenMP)
## Overview
This folder contains the code and implementation for Assignment 3 of the ECE 759 - High-Performance Computing course. The assignment focuses on implementing and parallelizing key computational algorithms using OpenMP, and analyzing their performance on large-scale datasets. The key tasks include:

- **Matrix Function**: Implemented a parallelized version of matrix multiplication based on the mmul2 algorithm from HW02. The task involves multiplying square matrices using multiple threads, measuring the performance for different thread counts.
- **2D Convolutions**: Implemented a parallelized version of 2D convolution using OpenMP, applied on a large square matrix. The task focuses on evaluating the performance improvement when increasing the number of threads.
- **Parallel Merge Sort Multiplication**: Implemented a merge sort algorithm parallelized using OpenMP tasks, measuring performance for varying thread counts and task scheduling thresholds.


## Analysis
*Documentation and methods for timing and ramdonization can be found in the root folder* 
### Matrix Multiplication
The matrix multiplication function (mmul) is a parallel version of the mmul2 function from HW02. The performance of the function is measured for square matrices of size n×n, with varying numbers of threads (t). The execution time is recorded for different values of t and plotted to observe the scaling behavior.
The task will be run as  

`` 
 ./task1 n t
``

n is a positive integer for the dimension of the matrices, t is an integer in the range [1, 20] for the number of threads  

#### Performance observation
The task is run with value n = 1024, and value t = 1,2,··· ,20.
![image](https://github.com/user-attachments/assets/27b308ca-1774-46e4-bfb3-35f823210760)

With increasing thread count, the time to complete the matrix multiplication decreases. However, after a certain number of threads, the performance improvement becomes less significant due to the overhead of thread management and cache effects.

### 2D Convolutions
The convolution function applies a 3×3 mask to a square matrix using parallel threads. The task involves testing different numbers of threads and measuring the time taken for the convolution operation.
The task will be run as  

`` 
 ./task2 n
``

where n is a positive integer for matrix dimension  

#### Performancse observation
The task it result of running n = 1024, and t = 1,2,··· ,20.
![image](https://github.com/user-attachments/assets/4799aee4-88f6-4f3c-bcb2-fb100e72d6c3)

- Initial Performance Gain: As the number of threads increases from 1 to around 8, there is a significant decrease in the time taken. This is expected, as parallelization allows more efficient processing of the matrix.
- Diminishing Returns: After a certain point (around 8 threads), the reduction in time becomes less significant. This could be due to the overhead of managing multiple threads, communication between them, and potential contention for shared resources.
- No Further Improvement: Beyond 16 threads, there is almost no reduction in execution time, and in some cases, the performance slightly degrades. This could be because the problem size may not be large enough to fully utilize all available threads, or due to the limitations of parallel scalability as the workload is distributed over too many threads.

### Parallel Merge Sort Multiplication
The merge sort algorithm was implemented using OpenMP tasks. The function sorts an array of random integers by dividing it into smaller parts and applying recursive merge sort in parallel.
The task will be run as

`` 
 ./task3 n t ts
``

where n is a positive integer for length of array, t is an integer in the range [1, 20] for number of threads, ts is the threshold as the lower limit to make recursive calls in order to avoid the overhead of recursion/task scheduling when the input array has small size; under this limit, a serial sorting algorithm without recursion calls will be used

#### Performancse observation

First plot: The task is run with n = 10^6, value t = 8, and value ts = 2^1,2^2,··· ,2^10.
![image](https://github.com/user-attachments/assets/b87a24d5-98f8-40a5-b1bb-87d10830c157)

- Result: As the threshold (ts) increases from 2^1 to 2^10, performance improves, with faster execution times observed at higher thresholds.
- Reduced Parallelism Overhead: Higher thresholds reduce the number of parallel tasks, lowering the overhead associated with creating and managing threads, leading to better performance.
- Optimal Task Size: Increasing the threshold results in more coarse-grained tasks, which are more efficient to process. Small tasks generate excessive overhead, while larger tasks strike a balance between parallelism and efficiency.
- Efficient Serial Sorting for Small Inputs: At higher thresholds, smaller subarrays are handled by serial sorting algorithms, which are faster and more cache-efficient for small data sizes, contributing to improved performance.
- Diminishing Returns: Performance gains from increasing the threshold stabilize after a certain point, as the benefits of reduced overhead and efficient serial sorting balance out with the loss of parallelism.

Second plot: The task is run with value n = 10^6, value t = 1,2,··· ,20, and ts = 256.
![image](https://github.com/user-attachments/assets/24a6da83-2f47-434c-a715-018b2d581054)

- Result: The performance improves significantly when increasing from 1 to 6 threads, but beyond 6 threads, the execution time shows smaller improvements, eventually plateauing between 10 and 20 threads.
- Initial Gain: The most dramatic performance improvement occurs between 1 and 6 threads, reducing execution time from 83.25 ms to 28.88 ms, highlighting the substantial benefit of parallelization during this range.
- Fluctuation Beyond 10 Threads: After 10 threads, adding more threads results in diminishing returns, with execution times stabilizing around 18-20 ms. The improvements beyond 10 threads are minimal and do not provide significant gains, indicating the overhead from managing additional threads.
- Possible Bottlenecks: The fluctuating performance could be attributed to factors like memory contention, synchronization overhead, or diminishing returns from parallelism, especially for smaller subarrays. As more threads are added beyond 10, the overhead of task scheduling and memory access contention begins to offset the gains from parallel execution.

## How to run
*These tasks can be run on non-euler machines.*
### Non Euler machines
After cloning the repo, cd to the deliverbles directory

`cd ../path_to_deliverbles`

OpenMP must be intalled. g++ that comes with MAC does not support openMP. Several libraries have to be installed with bres. See this [post](https://stackoverflow.com/questions/60005176/how-to-deal-with-clang-error-unsupported-option-fopenmp-on-travis) for more information.

The command I used to compile is (using task1 as an example):

``
/opt/homebrew/opt/llvm/bin/clang++ matmul.cpp task1.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp -isysroot $(xcrun --show-sdk-path) -L/opt/homebrew/opt/llvm/lib -L$(xcrun --show-sdk-path)/usr/lib -stdlib=libc++
``


### Euler machines
Run the shell scripts directly    

`./task1.sh`
