#include "vscale.cuh"
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: ./task3 n, where n is the length of the array";
        return 1;
    }

    // Parse the arguments
    size_t n = std::atoi(argv[1]);

    // Create two arrays of length n filled with random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist1(0.0f, 1.0f);


    // Allocate memory for host and device arrays
    float *a, *b, *d_a, *d_b;
    if (cudaMallocHost(&a, n * sizeof(float)) != cudaSuccess) {
        std::cerr << "Error allocating pinned memory for array a on host\n";
        return 1;
    }
    if (cudaMallocHost(&b, n * sizeof(float)) != cudaSuccess) {
        std::cerr << "Error allocating pinned memory for array b on host\n";
        cudaFreeHost(a);  // Free previously allocated memory
        return 1;
    }
    if (cudaMalloc((void**)&d_a, n * sizeof(float)) != cudaSuccess) {
        std::cerr << "Error allocating memory for array d_a on device\n";
        cudaFreeHost(a);
        cudaFreeHost(b);
        return 1;
    }
    if (cudaMalloc((void**)&d_b, n * sizeof(float)) != cudaSuccess) {
        std::cerr << "Error allocating memory for array d_b on device\n";
        cudaFreeHost(a);
        cudaFreeHost(b);
        cudaFree(d_a);
        return 1;
    }

    // Fill host arrays with random values
    for (size_t i = 0; i < n; ++i) {
        a[i] = dist(gen);
        b[i] = dist1(gen);
    }

    // Copy data from host to device
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel execution configuration
    int numThreadsPerBlock = 16;
    int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call the kernel
    vscale<<<numBlocks, numThreadsPerBlock>>>(d_a, d_b, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Print the amount of time taken to execute the kernel in milliseconds
    printf("%f\n", ms);

    // Copy the result from device back to host
    cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first and last elements of the resulting array
    printf("%f\n", b[0]);
    printf("%f\n", b[n - 1]);
    printf("\n");

    // Free device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFreeHost(a);
    cudaFreeHost(b);

    return 0;
}
