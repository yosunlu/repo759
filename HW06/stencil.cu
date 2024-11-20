#include "stencil.cuh"
#include <iostream>
#include <cstdlib>
#include <cstdio>

// Computes the convolution of image and mask, storing the result in output.
// Each thread should compute _one_ element of the output matrix.
// Shared memory should be allocated _dynamically_ only.
//
// image is an array of length n.
// mask is an array of length (2 * R + 1).
// output is an array of length n.
// All of them are in device memory
//
// Assumptions:
// - 1D configuration
// - blockDim.x >= 2 * R + 1
//
// The following should be stored/computed in shared memory:
// - The entire mask
// - The elements of image that are needed to compute the elements of output corresponding to the threads in the given block
// - The output image elements corresponding to the given block before it is written back to global memory
__global__ void stencil_kernel(const float *image, const float *mask, float *output, unsigned int n, unsigned int R)
{
    // Calculate the thread index in the global matrix
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory allocation for mask and the image block
    extern __shared__ float shared_mem[];
    float *shared_mask = shared_mem;
    float *shared_image = &shared_mem[2 * R + 1];

    // Thread-local index
    int local_idx = threadIdx.x;

    int block_idx = blockIdx.x;

    // Load mask into shared memory (handled by the first threads)
    // say R == 2, then mask has 2 * 2 + 1 elements; only need the first 5 threads of the block to load mask from global to shared
    if (local_idx < 2 * R + 1)
    {
        shared_mask[local_idx] = mask[local_idx];
    }
    printf("currently in block:%d, local thread: %d, global thread: %d, mask: %d\n", block_idx, local_idx, global_idx, shared_mask[local_idx]);

    // Load corresponding image elements into shared memory
    // say blockDim == 5 (each block has 7 threads) and R == 2; halo_left for the threads in 0th block is -2
    // int halo_left = blockIdx.x * blockDim.x - R;
    // // for the first 2 threads in the 0th bock, shared_image_idx < 0
    // int shared_image_idx = halo_left + local_idx;

    // if (shared_image_idx < 0 || shared_image_idx >= n)
    // {
    //     // Out-of-bound pixels are assumed to be 1
    //     shared_image[local_idx] = 1.0f;
    // }
    // else
    // {
    //     shared_image[local_idx] = image[shared_image_idx];
    // }

    // // Synchronize all threads in the block after loading shared memory
    // __syncthreads();

    // // Compute the stencil operation for this thread if it's within bounds
    // if (global_idx < n)
    // {
    //     float result = 0.0f;
    //     for (int j = -R; j <= R; ++j)
    //     {
    //         int mask_idx = j + R; // Adjust for mask indexing
    //         int shared_image_idx = local_idx + j;

    //         result += shared_image[shared_image_idx] * shared_mask[mask_idx];
    //     }

    //     // Write the result to global memory
    //     output[global_idx] = result;
    // }

    // // Synchronize again to ensure all threads finish before returning
    __syncthreads();
}

// Makes one call to stencil_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
//
// Assumptions:
// - threads_per_block >= 2 * R + 1
__host__ void stencil(const float *image,
                      const float *mask,
                      float *output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block)
{
    // Compute the number of blocks needed based on the threads per block
    unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    // Calculate size of shared memory
    // (2 * R + 1) * sizeof(float) is the size of the mask
    // (threads_per_block + 2 * R) is the element needed 
    unsigned int shared_size = (2 * R + 1) * sizeof(float) + (threads_per_block + 2 * R) * sizeof(float);
    for (int i = 0; i < 5; ++i)
    {
        std::cout << mask[i] << std::endl; 
    }

    // Launch the kernel with the calculated configuration
    stencil_kernel<<<num_blocks, threads_per_block, shared_size>>>(image, mask, output, n, R);

    // Synchronize after kernel execution
    cudaDeviceSynchronize();
}
