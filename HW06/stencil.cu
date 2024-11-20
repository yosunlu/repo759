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
    // devide the shared memory array
    float *shared_mask = shared_mem;
    float *shared_image = &shared_mem[2 * R + 1];

    int local_idx = threadIdx.x;
    int block_idx = blockIdx.x;

    // Load mask into shared memory (handled by the first threads)
    if (local_idx < 2 * R + 1)
    {
        shared_mask[local_idx] = mask[local_idx];
    }
    // printf("currently in block: %d, local thread: %d, global thread: %d, mask: %f\n", block_idx, local_idx, global_idx, shared_mask[local_idx]);
    
    // Load corresponding image elements into shared memory
    // load left halo
    if (local_idx == 0){
        int shared_image_idx = 0;
        for(int i = R; i > 0; --i){
            int left_idx = global_idx - i;
            shared_image[shared_image_idx] = left_idx < 0 ? 1.0f : image[left_idx];
            // if(blockIdx.x == 1) printf("block: %d, shared_image_idx[%d]: %f\n", block_idx, shared_image_idx, shared_image[shared_image_idx]);
            shared_image_idx++;
        }
    }

    // load right halo
    if (local_idx == blockDim.x - 1){
        int shared_image_idx = blockDim.x + R;
        for(int i = 1; i <= R; ++i){
            int right_idx = global_idx + i;
            shared_image[shared_image_idx] = right_idx >= n ? 1.0f : image[right_idx];
            // if(blockIdx.x == 1) printf("block: %d, shared_image_idx[%d]: %f\n", block_idx, shared_image_idx, shared_image[shared_image_idx]);
            shared_image_idx++;
        }
    }

    shared_image[R + local_idx] = image[global_idx];
    // if(blockIdx.x == 1) printf("block: %d, shared_image_idx[%d]: %f\n", block_idx, R + local_idx, shared_image[R + local_idx]);
     __syncthreads();



    // Compute the stencil operation for this thread if it's within bounds
    float result = 0.0f;
    
    for (int j = -2; j <= 2; ++j)
    {
        // printf("debug");
        int mask_idx = j + R; // Adjust for mask indexing
        int shared_image_idx = R + local_idx + j;

        result += shared_image[shared_image_idx] * shared_mask[mask_idx];
        if (global_idx == 5){
            printf("j: %d, shared_image_idx: %d, shared_image[%d]: %f, mask: %f\n", j, shared_image_idx, shared_image_idx, shared_image[shared_image_idx], shared_mask[mask_idx]);
        }
    }

    // Write the result to global memory
    output[global_idx] = result;

    // Synchronize again to ensure all threads finish before returning
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

    // Launch the kernel with the calculated configuration
    stencil_kernel<<<num_blocks, threads_per_block, shared_size>>>(image, mask, output, n, R);

    // Synchronize after kernel execution
    cudaDeviceSynchronize();
}

