#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "matmul.cuh"

// Kernel for tiled matrix multiplication using dynamic shared memory for int
__global__ void matmul_kernel_int(const int *A, const int *B, int *C, unsigned int n)
{
    // Dynamic shared memory for tiles of A and B
    extern __shared__ int shared_memory[];
    int *shared_A = shared_memory;                             // First half for A
    int *shared_B = (int *)&shared_A[blockDim.y * blockDim.x]; // Second half for B

    // Block and thread indices
    int bx = blockIdx.x;  // Block column index // 0
    int by = blockIdx.y;  // Block row index // 1
    int tx = threadIdx.x; // Thread column index // 1
    int ty = threadIdx.y; // Thread row index // 0

    // Compute global row and column for matrix C
    int row = by * blockDim.y + ty; // 2
    int col = bx * blockDim.x + tx; // 1

    int Csub = 0; // Accumulator for this thread's result

    // Loop over tiles of A and B
    for (int t = 0; t < (n + blockDim.x - 1) / blockDim.x; ++t)
    { // t = 0, 1
        // Row and column indices for current tile
        int aRow = row;                 // 2
        int aCol = t * blockDim.x + tx; // 1
        int bRow = t * blockDim.y + ty; // 0
        int bCol = col;                 // 1

        // Load tiles into shared memory with boundary checks
        if (aRow < n && aCol < n)
        {
            shared_A[ty * blockDim.x + tx] = A[aRow * n + aCol]; // shared_A[1] = A[9]
        }
        else
        {
            shared_A[ty * blockDim.x + tx] = 0; // Pad with zeros
        }

        if (bRow < n && bCol < n)
        {
            shared_B[ty * blockDim.x + tx] = B[bRow * n + bCol]; // shared_B[1] = B[1]
        }
        else
        {
            shared_B[ty * blockDim.x + tx] = 0; // Pad with zeros
        }

        // Synchronize threads to ensure all tiles are loaded
        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < blockDim.x; ++k)
        {
            Csub += shared_A[ty * blockDim.x + k] * shared_B[k * blockDim.x + tx]; // shared_A[0] * shared_B[1]
        }

        // Synchronize threads before loading the next tile
        __syncthreads();
    }

    // Write the computed value to global memory with boundary check
    if (row < n && col < n)
    {
        C[row * n + col] = Csub;
    }
}

// Kernel for tiled matrix multiplication using dynamic shared memory for float
__global__ void matmul_kernel_float(const float *A, const float *B, float *C, unsigned int n)
{
    // Dynamic shared memory for tiles of A and B
    extern __shared__ float float_shared_memory[];
    float *shared_A = float_shared_memory;                               // First half for A
    float *shared_B = (float *)&shared_A[blockDim.y * blockDim.x]; // Second half for B

    // Block and thread indices
    int bx = blockIdx.x;  // Block column index // 0
    int by = blockIdx.y;  // Block row index // 1
    int tx = threadIdx.x; // Thread column index // 1
    int ty = threadIdx.y; // Thread row index // 0

    // Compute global row and column for matrix C
    int row = by * blockDim.y + ty; // 2
    int col = bx * blockDim.x + tx; // 1

    int Csub = 0.0f; // Accumulator for this thread's result

    // Loop over tiles of A and B
    for (int t = 0; t < (n + blockDim.x - 1) / blockDim.x; ++t)
    { // t = 0, 1
        // Row and column indices for current tile
        int aRow = row;                 // 2
        int aCol = t * blockDim.x + tx; // 1
        int bRow = t * blockDim.y + ty; // 0
        int bCol = col;                 // 1

        // Load tiles into shared memory with boundary checks
        if (aRow < n && aCol < n)
        {
            shared_A[ty * blockDim.x + tx] = A[aRow * n + aCol]; // shared_A[1] = A[9]
        }
        else
        {
            shared_A[ty * blockDim.x + tx] = 0.0f; // Pad with zeros
        }

        if (bRow < n && bCol < n)
        {
            shared_B[ty * blockDim.x + tx] = B[bRow * n + bCol]; // shared_B[1] = B[1]
        }
        else
        {
            shared_B[ty * blockDim.x + tx] = 0.0f; // Pad with zeros
        }

        // Synchronize threads to ensure all tiles are loaded
        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < blockDim.x; ++k)
        {
            Csub += shared_A[ty * blockDim.x + k] * shared_B[k * blockDim.x + tx]; // shared_A[0] * shared_B[1]
        }

        // Synchronize threads before loading the next tile
        __syncthreads();
    }

    // Write the computed value to global memory with boundary check
    if (row < n && col < n)
    {
        C[row * n + col] = Csub;
    }
}

// Kernel for tiled matrix multiplication using dynamic shared memory for double
__global__ void matmul_kernel_double(const double *A, const double *B, double *C, unsigned int n)
{
    // Dynamic shared memory for tiles of A and B
    extern __shared__ double double_shared_memory[];
    double *shared_A = double_shared_memory;                                // First half for A
    double *shared_B = (double *)&shared_A[blockDim.y * blockDim.x]; // Second half for B

    // Block and thread indices
    int bx = blockIdx.x;  // Block column index
    int by = blockIdx.y;  // Block row index
    int tx = threadIdx.x; // Thread column index
    int ty = threadIdx.y; // Thread row index

    // Compute global row and column for matrix C
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    double Csub = 0.0; // Accumulator for this thread's result

    // Loop over tiles of A and B
    for (int t = 0; t < (n + blockDim.x - 1) / blockDim.x; ++t)
    {
        // Load tiles into shared memory with boundary checks
        if (row < n && t * blockDim.x + tx < n)
            shared_A[ty * blockDim.x + tx] = A[row * n + t * blockDim.x + tx];
        else
            shared_A[ty * blockDim.x + tx] = 0.0;

        if (t * blockDim.y + ty < n && col < n)
            shared_B[ty * blockDim.x + tx] = B[(t * blockDim.y + ty) * n + col];
        else
            shared_B[ty * blockDim.x + tx] = 0.0;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < blockDim.x; ++k)
        {
            Csub += shared_A[ty * blockDim.x + k] * shared_B[k * blockDim.x + tx];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < n && col < n)
    {
        C[row * n + col] = Csub;
    }
}

// Host function to invoke the kernel for int
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim)
{
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

// Host function to invoke the kernel for float
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim)
{
    int blockNum = (n + block_dim - 1) / block_dim;

    // Define grid and block dimensions
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(blockNum, blockNum);

    // Compute shared memory size (for two tiles)
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(float);

    // Launch the kernel
    matmul_kernel_float<<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n);

    // Synchronize device
    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim)
{
    int blockNum = (n + block_dim - 1) / block_dim;

    // Define grid and block dimensions
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(blockNum, blockNum);

    // Compute shared memory size (for two tiles)
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(double);

    // Launch the kernel
    matmul_kernel_double<<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n);

    // Synchronize device
    cudaDeviceSynchronize();
}