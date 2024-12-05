#include "matmul.cuh"
#include <cuda.h>
#include <iostream>
#include <random>
#include <cuda_runtime.h>

// int run_matmul_int(unsigned int n, unsigned int size, unsigned int block_dim)
// {
//     // generate random variables
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<int> dist(-10, 10);

//     // Allocate memory for host and device arrays
//     int *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
//     if (cudaMallocHost(&h_a, size * sizeof(int)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating pinned memory for array h_a on host\n";
//         return 1;
//     }
//     if (cudaMallocHost(&h_b, size * sizeof(int)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating pinned memory for array h_b on host\n";
//         cudaFreeHost(h_a); // Free previously allocated memory
//         return 1;
//     }
//     if (cudaMallocHost(&h_c, size * sizeof(int)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating pinned memory for array h_b on host\n";
//         cudaFreeHost(h_a); // Free previously allocated memory
//         cudaFreeHost(h_b);
//         return 1;
//     }
//     if (cudaMalloc((void **)&d_a, size * sizeof(int)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating memory for array d_a on device\n";
//         cudaFreeHost(h_a);
//         cudaFreeHost(h_b);
//         cudaFreeHost(h_c);
//         return 1;
//     }
//     if (cudaMalloc((void **)&d_b, size * sizeof(int)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating memory for array d_b on device\n";
//         cudaFreeHost(h_a);
//         cudaFreeHost(h_b);
//         cudaFreeHost(h_c);
//         cudaFree(d_a);
//         return 1;
//     }
//     if (cudaMalloc((void **)&d_c, size * sizeof(int)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating memory for array d_c on device\n";
//         cudaFreeHost(h_a);
//         cudaFreeHost(h_b);
//         cudaFreeHost(h_c);
//         cudaFree(d_a);
//         cudaFree(d_b);
//         return 1;
//     }

//     // Fill host arrays with random values
//     for (size_t i = 0; i < size; ++i)
//     {
//         h_a[i] = dist(gen);
//         h_b[i] = dist(gen);
//     }

//     // Copy data from host to device
//     cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

//     // Set up CUDA events for timing
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

//     // Run tests for matmul_1, matmul_2, and matmul_3
//     matmul_1(d_a, d_b, d_c, n, block_dim);

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     // Get the elapsed time in milliseconds
//     float ms;
//     cudaEventElapsedTime(&ms, start, stop);

//     // Copy the result from device back to host
//     cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

//     // Print the last element of the resulting matrix.
//     printf("first element: %d\n", h_c[0]);

//     // Print the last element of the resulting matrix.
//     printf("last element: %d\n", h_c[size - 1]);

//     // Print the amount of time taken to execute the kernel in milliseconds
//     printf("time taken: %f\n", ms);

//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
//     cudaFreeHost(h_a);
//     cudaFreeHost(h_b);
//     cudaFreeHost(h_c);

//     return 0;
// }

// int run_matmul_float(unsigned int n, unsigned int size, unsigned int block_dim)
// {
//     // Generate random variables
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

//     // Allocate memory for host and device arrays
//     float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
//     if (cudaMallocHost(&h_a, size * sizeof(float)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating pinned memory for array h_a on host\n";
//         return 1;
//     }
//     if (cudaMallocHost(&h_b, size * sizeof(float)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating pinned memory for array h_b on host\n";
//         cudaFreeHost(h_a);
//         return 1;
//     }
//     if (cudaMallocHost(&h_c, size * sizeof(float)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating pinned memory for array h_c on host\n";
//         cudaFreeHost(h_a);
//         cudaFreeHost(h_b);
//         return 1;
//     }
//     if (cudaMalloc((void **)&d_a, size * sizeof(float)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating memory for array d_a on device\n";
//         cudaFreeHost(h_a);
//         cudaFreeHost(h_b);
//         cudaFreeHost(h_c);
//         return 1;
//     }
//     if (cudaMalloc((void **)&d_b, size * sizeof(float)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating memory for array d_b on device\n";
//         cudaFreeHost(h_a);
//         cudaFreeHost(h_b);
//         cudaFreeHost(h_c);
//         cudaFree(d_a);
//         return 1;
//     }
//     if (cudaMalloc((void **)&d_c, size * sizeof(float)) != cudaSuccess)
//     {
//         std::cerr << "Error allocating memory for array d_c on device\n";
//         cudaFreeHost(h_a);
//         cudaFreeHost(h_b);
//         cudaFreeHost(h_c);
//         cudaFree(d_a);
//         cudaFree(d_b);
//         return 1;
//     }

//     // Fill host arrays with random values
//     for (size_t i = 0; i < size; ++i)
//     {
//         h_a[i] = dist(gen);
//         h_b[i] = dist(gen);
//     }

//     // Copy data from host to device
//     cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

//     // Set up CUDA events for timing
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

//     // Run the kernel
//     matmul_2(d_a, d_b, d_c, n, block_dim);

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     // Get the elapsed time in milliseconds
//     float ms;
//     cudaEventElapsedTime(&ms, start, stop);

//     // Copy the result from device back to host
//     cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

//     // Print the first and last elements of the resulting matrix
//     printf("First element: %f\n", h_a[0]);
//     printf("Last element: %f\n", h_a[size - 1]);

//     // Print the amount of time taken to execute the kernel
//     printf("Time taken: %f ms\n", ms);

//     // Cleanup
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
//     cudaFreeHost(h_a);
//     cudaFreeHost(h_b);
//     cudaFreeHost(h_c);

//     return 0;
// }

int run_matmul_double(unsigned int n, unsigned int size, unsigned int block_dim)
{
    // Generate random variables
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    // Allocate memory for host and device arrays
    double *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
    if (cudaMallocHost(&h_a, size * sizeof(double)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for array h_a on host\n";
        return 1;
    }
    if (cudaMallocHost(&h_b, size * sizeof(double)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for array h_b on host\n";
        cudaFreeHost(h_a);
        return 1;
    }
    if (cudaMallocHost(&h_c, size * sizeof(double)) != cudaSuccess)
    {
        std::cerr << "Error allocating pinned memory for array h_c on host\n";
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        return 1;
    }
    if (cudaMalloc((void **)&d_a, size * sizeof(double)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_a on device\n";
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return 1;
    }
    if (cudaMalloc((void **)&d_b, size * sizeof(double)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_b on device\n";
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        cudaFree(d_a);
        return 1;
    }
    if (cudaMalloc((void **)&d_c, size * sizeof(double)) != cudaSuccess)
    {
        std::cerr << "Error allocating memory for array d_c on device\n";
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
        h_b[i] = dist(gen);
    }

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(double), cudaMemcpyHostToDevice);

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run the kernel
    matmul_3(d_a, d_b, d_c, n, block_dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy the result from device back to host
    cudaMemcpy(h_c, d_c, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the first and last elements of the resulting matrix
    printf("First element: %f\n", h_a[0]);
    printf("Last element: %f\n", h_a[size - 1]);

    // Print the amount of time taken to execute the kernel
    printf("Time taken: %f ms\n", ms);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: ./task1 n block_dim\n";
        return 1;
    }

    // Parse arguments
    unsigned int n = std::atoi(argv[1]);
    unsigned int size = n * n;
    unsigned int block_dim = std::atoi(argv[2]);

    // run_matmul_int(n, size, block_dim);
    // run_matmul_float(n, size, block_dim);
    run_matmul_double(n, size, block_dim);

    return 0;
}
