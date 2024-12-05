#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "matmul.cuh"

// Kernel for tiled matrix multiplication using dynamic shared memory
__global__ void matmul_kernel_int(const int* A, const int* B, int* C, unsigned int n) {
    // Dynamic shared memory for tiles of A and B
    extern __shared__ int shared_memory[];
    int *shared_A = shared_memory; // First half for A
    int *shared_B = (int *)&shared_A[blockDim.y * blockDim.x]; // Second half for B

    // Block and thread indices
    int bx = blockIdx.x;  // Block column index
    int by = blockIdx.y;  // Block row index
    int tx = threadIdx.x; // Thread column index
    int ty = threadIdx.y; // Thread row index

    // Compute global row and column for matrix C
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int Csub = 0; // Accumulator for this thread's result

    // Loop over tiles of A and B
    for (int t = 0; t < (n + blockDim.x - 1) / blockDim.x; ++t) {
        // Row and column indices for current tile
        int aRow = row;
        int aCol = t * blockDim.x + tx;
        int bRow = t * blockDim.y + ty;
        int bCol = col;

        // Load tiles into shared memory with boundary checks
        if (aRow < n && aCol < n) {
            shared_A[ty * blockDim.x + tx] = A[aRow * n + aCol];
        } else {
            shared_A[ty * blockDim.x + tx] = 0; // Pad with zeros
        }

        if (bRow < n && bCol < n) {
            shared_B[ty * blockDim.x + tx] = B[bRow * n + bCol];
        } else {
            shared_B[ty * blockDim.x + tx] = 0; // Pad with zeros
        }

        // Synchronize threads to ensure all tiles are loaded
        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < blockDim.x; ++k) {
            Csub += shared_A[ty * blockDim.x + k] * shared_B[k * blockDim.x + tx];
        }

        // Synchronize threads before loading the next tile
        __syncthreads();
    }

    // Write the computed value to global memory with boundary check
    if (row < n && col < n) {
        C[row * n + col] = Csub;
    }
}

// Host function to invoke the kernel
__host__ void matmul_1(const int* A, const int* B, int* C, unsigned int n, unsigned int block_dim) {
    int blockNum = (n + block_dim - 1) / block_dim;

    // Define grid and block dimensions
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(blockNum, blockNum);

    // Compute shared memory size (for two tiles)
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(int);

    // Launch the kernel
    matmul_kernel_int<<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n);

    // Synchronize device
    cudaDeviceSynchronize();
}
