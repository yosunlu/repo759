// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float *A, const float *B, float *C, size_t n)
{
    // Calculate the thread index in the flattened matrix
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure the thread index is within bounds
    if (idx < n * n)
    {
        int row = idx / n;   // Row index in the result matrix
        int col = idx % n;   // Column index in the result matrix

        // Perform dot product of the row from A and the column from B
        float sum = 0;
        for (int i = 0; i < n; ++i)
        {
            sum += A[row * n + i] * B[i * n + col];
        }

        // Store the computed value in the result matrix
        C[idx] = sum;
    }
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float *A, const float *B, float *C, size_t n, unsigned int threads_per_block)
{
    // Calculate the total number of threads required for an n x n matrix
    unsigned int total_threads = n * n;

    // Compute the number of blocks needed based on the threads per block
    unsigned int numBlocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Launch the kernel with the calculated configuration
    matmul_kernel<<<numBlocks, threads_per_block>>>(A, B, C, n);

    // Wait for the device to finish execution (if required for debugging/timing)
    cudaDeviceSynchronize();
}
