#include "matmul.cuh"
#include <cuda.h>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{

    if (argc != 3)
    {
        std::cerr << "Usage: ./task1 n threads_per_block, where n is the length of the array";
        return 1;
    }

    // Parse the arguments
    size_t n = std::atoi(argv[1]);
    size_t size = n * n;
    size_t threads_per_block = std::atoi(argv[2]);

    // Create two arrays of length n filled with random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Allocate memory for host and device arrays
    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
    if (cudaMallocHost(&h_a, size * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for array h_a on host\n";
        return 1;
    }
    if (cudaMallocHost(&h_b, size * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for array h_b on host\n";
        cudaFreeHost(h_a); // Free previously allocated memory
        return 1;
    }
    if (cudaMallocHost(&h_c, size * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for array h_b on host\n";
        cudaFreeHost(h_a); // Free previously allocated memory
        cudaFreeHost(h_b);
        return 1;
    }
    if (cudaMalloc((void **)&d_a, size * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_a on device\n";
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return 1;
    }
    if (cudaMalloc((void **)&d_b, size * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_b on device\n";
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        cudaFree(d_a);
        return 1;
    }
    if (cudaMalloc((void **)&d_c, size * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_b on device\n";
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        cudaFree(d_a);
        cudaFree(d_b);
        return 1;
    }

    // Fill host arrays with random values
    for (size_t i = 0; i < size; ++i)
    {
        h_a[i] = dist(gen);
        std::cout << h_a[i] << std::endl; 
        h_b[i] = dist(gen);
    }

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // call the matmul function
    matmul(d_a, d_b, d_c, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy the result from device back to host
    cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the last element of the resulting matrix.
    printf("%f\n", h_c[size - 1]);

    // Print the amount of time taken to execute the kernel in milliseconds
    printf("time taken: %f\n", ms);

    // Free device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
}
