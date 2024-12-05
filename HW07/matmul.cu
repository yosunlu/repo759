#include "matmul.cuh"

#define TILE_SIZE 16

// Kernel for Tiled Matrix Multiplication
template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, unsigned int n) {
    // Block and thread indices
    unsigned int bx = blockIdx.x; // B (and C) tile column index
    unsigned int by = blockIdx.y; // A (and C) tile row index
    unsigned int tx = threadIdx.x; // Tile column index
    unsigned int ty = threadIdx.y; // Tile row index

    // Sub-matrix C element computed by this thread
    T Csub = 0;

    // Shared memory tiles for A and B
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];

    // Iterate over all tiles required to compute Csub
    for (unsigned int phase = 0; phase < (n + TILE_SIZE - 1) / TILE_SIZE; ++phase) {
        // Load the A and B tiles into shared memory
        unsigned int aRow = by * TILE_SIZE + ty;
        unsigned int aCol = phase * TILE_SIZE + tx;
        unsigned int bRow = phase * TILE_SIZE + ty;
        unsigned int bCol = bx * TILE_SIZE + tx;

        if (aRow < n && aCol < n) {
            As[ty][tx] = A[aRow * n + aCol];
        } else {
            As[ty][tx] = 0;
        }

        if (bRow < n && bCol < n) {
            Bs[ty][tx] = B[bRow * n + bCol];
        } else {
            Bs[ty][tx] = 0;
        }

        // Synchronize threads to ensure all elements are loaded
        __syncthreads();

        // Compute partial product for this tile
        for (unsigned int k = 0; k < TILE_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize threads before loading the next tiles
        __syncthreads();
    }

    // Write the computed value to C
    unsigned int cRow = by * TILE_SIZE + ty;
    unsigned int cCol = bx * TILE_SIZE + tx;
    if (cRow < n && cCol < n) {
        C[cRow * n + cCol] = Csub;
    }
}

// Host functions to invoke the kernel
template <typename T>
__host__ void matmul_template(const T* A, const T* B, T* C, unsigned int n, unsigned int block_dim) {
    // Configure grid and block dimensions
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);

    // Launch kernel
    matmul_kernel<<<dimGrid, dimBlock>>>(A, B, C, n);

    // Synchronize device
    cudaDeviceSynchronize();
}

__host__ void matmul_1(const int* A, const int* B, int* C, unsigned int n, unsigned int block_dim) {
    matmul_template<int>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float* A, const float* B, float* C, unsigned int n, unsigned int block_dim) {
    matmul_template<float>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double* A, const double* B, double* C, unsigned int n, unsigned int block_dim) {
    matmul_template<double>(A, B, C, n, block_dim);
}
