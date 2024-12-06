#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n)
{
    // Dynamically allocated shared memory for intermediate sums
    extern __shared__ float sdata[];

    // Thread index within the block
    unsigned int tid = threadIdx.x;

    // Index in the global array for this thread
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Perform the first level of reduction during global load
    // Load two elements from global memory and add them, if within bounds
    if (i < n) {
        sdata[tid] = g_idata[i] + ((i + blockDim.x < n) ? g_idata[i + blockDim.x] : 0.0f);
    } else {
        sdata[tid] = 0.0f; // Pad with zeros for out-of-bound threads
    }

    // Synchronize threads to ensure shared memory is fully populated
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result from this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block)
{
    unsigned int blocks_per_grid = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

    // Allocate dynamically shared memory (one element per thread)
    size_t shared_mem_size = threads_per_block * sizeof(float);

    // Perform multiple reductions until the result fits in one block
    while (blocks_per_grid > 1) {
        reduce_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            *input, *output, N);

        cudaDeviceSynchronize();

        // Swap input and output pointers for the next iteration
        std::swap(*input, *output);

        // Update N and blocks_per_grid for the next round
        N = blocks_per_grid;
        blocks_per_grid = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    }

    // Perform the final reduction if needed
    reduce_kernel<<<1, threads_per_block, shared_mem_size>>>(*input, *output, N);

    cudaDeviceSynchronize();
}
