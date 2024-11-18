#include "stencil.cuh"

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
    // Calculate the thread index in the flattened matrix
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    extern __shared__ float Ms[];

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
    unsigned int numBlocks = (n + threads_per_block - 1) / threads_per_block;

    // Calculate size of shared memory
    unsigned int shareSize = (2 * R + 1) * sizeof(float) + threads_per_block * sizeof(float);

    // Launch the kernel with the calculated configuration
    stencil_kernel<<<numBlocks, threads_per_block, shareSize>>>(image, mask, output, n, R);
}