#include "stencil.cuh"
#include <cuda.h>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{

    // Allocate memory for host and device arrays
    float *h_i, *h_m, *h_o, *d_i, *d_m, *d_o;

    if (cudaMallocHost(&h_i, 10 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for array h_i on host\n";
        return 1;
    }

    if (cudaMallocHost(&h_m, 5 * sizeof(float)) != cudaSuccess)
    {
        cudaFreeHost(h_i); // Free previously allocated memory
        std::cerr << "Error allocating pinned memory for array h_m on host\n";
        return 1;
    }

    if (cudaMallocHost(&h_o, 10 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for array h_b on host\n";
        cudaFreeHost(h_i); // Free previously allocated memory
        cudaFreeHost(h_m);
        return 1;
    }

    if (cudaMalloc((void **)&d_i, 10 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_i on device\n";
        cudaFreeHost(h_i);
        cudaFreeHost(h_m);
        cudaFreeHost(h_o);
        return 1;
    }

    if (cudaMalloc((void **)&d_m, 5 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_m on device\n";
        cudaFreeHost(h_i);
        cudaFreeHost(h_m);
        cudaFreeHost(h_o);
        cudaFree(d_i);
        return 1;
    }

    if (cudaMalloc((void **)&d_o, 10 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_o on device\n";
        cudaFreeHost(h_i);
        cudaFreeHost(h_m);
        cudaFreeHost(h_o);
        cudaFree(d_i);
        cudaFree(d_m);
        return 1;
    }

    // Fill host image with  values
    for (size_t i = 0; i < 10; ++i)
    {
        h_i[i] = i;
    }

    // Fill host mask with  values
    for (int i = 0; i < 5; ++i)
    {
        h_m[i] = -1 * i; 
        // std::cout << h_m[i] << std::endl; 
    }

    // Copy data from host to device
    cudaMemcpy(d_i, h_i, 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, 5 * sizeof(float), cudaMemcpyHostToDevice);

    // call the stencil function
    stencil(d_i, d_m, d_o, 10, 2, 5);

    // Copy data from device back to host
    cudaMemcpy(h_o, d_o, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i)
    {
        std::cout << h_o[i] << std::endl;
    }

    // Free device and host memory
    cudaFree(d_i);
    cudaFree(d_m);
    cudaFree(d_o);
    cudaFreeHost(h_i);
    cudaFreeHost(h_m);
    cudaFreeHost(h_o);


}