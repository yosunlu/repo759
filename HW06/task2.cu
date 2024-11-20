#include "stencil.cuh"
#include <cuda.h>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        std::cerr << "Usage: ./task2 n R threads_per_block, where 2 * R +1 is the length of the mask, and n is the length of the array";
        return 1;
    }

    // Parse the arguments
    size_t n = std::atoi(argv[1]); 
    unsigned int R = std::atoi(argv[2]);;
    size_t threads_per_block = std::atoi(argv[3]);

    // Allocate memory for host and device arrays
    float *h_i, *h_m, *h_o, *d_i, *d_m, *d_o;

    if (cudaMallocHost(&h_i, n * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for array h_i on host\n";
        return 1;
    }

    if (cudaMallocHost(&h_m, (2 * R + 1) * sizeof(float)) != cudaSuccess)
    {
        cudaFreeHost(h_i); // Free previously allocated memory
        std::cerr << "Error allocating pinned memory for array h_m on host\n";
        return 1;
    }

    if (cudaMallocHost(&h_o, n * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for array h_b on host\n";
        cudaFreeHost(h_i); // Free previously allocated memory
        cudaFreeHost(h_m);
        return 1;
    }

    if (cudaMalloc((void **)&d_i, n * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_i on device\n";
        cudaFreeHost(h_i);
        cudaFreeHost(h_m);
        cudaFreeHost(h_o);
        return 1;
    }

    if (cudaMalloc((void **)&d_m, (2 * R + 1) * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_m on device\n";
        cudaFreeHost(h_i);
        cudaFreeHost(h_m);
        cudaFreeHost(h_o);
        cudaFree(d_i);
        return 1;
    }

    if (cudaMalloc((void **)&d_o, n * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_o on device\n";
        cudaFreeHost(h_i);
        cudaFreeHost(h_m);
        cudaFreeHost(h_o);
        cudaFree(d_i);
        cudaFree(d_m);
        return 1;
    }

    // Fill the image and mask array with random numbers in the range [-1, 1].
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Fill host image with  values
    for (size_t i = 0; i < n; ++i)
    {
        h_i[i] = dist(gen);
    }

    // Fill host mask with  values
    for (int i = 0; i < (int)R; ++i)
    {
        h_m[i] = dist(gen); 
    }

    // Copy data from host to device
    cudaMemcpy(d_i, h_i, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);


    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // call the stencil function
    stencil(d_i, d_m, d_o, n, R, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy data from device back to host
    cudaMemcpy(h_o, d_o, n * sizeof(float), cudaMemcpyDeviceToHost);
 
    // Print the last element of the output matrix.
    printf("%f\n", h_o[n - 1]);

    // Print the amount of time taken to execute the kernel in milliseconds
    printf("time taken: %f\n", ms);

    // Free device and host memory
    cudaFree(d_i);
    cudaFree(d_m);
    cudaFree(d_o);
    cudaFreeHost(h_i);
    cudaFreeHost(h_m);
    cudaFreeHost(h_o);

}