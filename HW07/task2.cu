#include <cuda.h>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "reduce.cuh"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: ./task2 N threads_per_block\n";
        return 1;
    }

    // Parse the arguments
    size_t N = std::atoi(argv[1]);
    size_t threads_per_block = std::atoi(argv[2]);

    // Allocate memory for host and device arrays
    float *h_input, *h_output, *d_input, *d_output;

    if (cudaMallocHost(&h_input, N * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for input array on host\n";
        return 1;
    }

    size_t num_blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

    if (cudaMallocHost(&h_output, num_blocks * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for output array on host\n";
        cudaFreeHost(h_input);
        return 1;
    }

    if (cudaMalloc((void **)&d_input, N * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for input array on device\n";
        cudaFreeHost(h_input);
        cudaFreeHost(h_output);
        return 1;
    }

    if (cudaMalloc((void **)&d_output, num_blocks * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for output array on device\n";
        cudaFreeHost(h_input);
        cudaFreeHost(h_output);
        cudaFree(d_input);
        return 1;
    }

    // Fill the input array with random numbers in the range [-1, 1].
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < N; ++i)
    {
        h_input[i] = dist(gen);
    }

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call the reduce function
    reduce(&d_input, &d_output, N, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy the result from device back to host
    cudaMemcpy(h_output, d_output, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the resulting sum
    printf("%f\n", h_output[0]);

    // Print the time taken
    printf("%f\n", ms);

    // Free device and host memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);

    return 0;
}