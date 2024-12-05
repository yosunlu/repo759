#include "matmul.cuh"
#include <cuda.h>
#include <iostream>
#include <random>
#include <cuda_runtime.h>

template <typename T>
void initialize_matrix(T *matrix, size_t size, T min_val, T max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min_val, max_val);

    for (size_t i = 0; i < size; ++i) {
        matrix[i] = static_cast<T>(dist(gen));
    }
}

template <typename T>
void run_matmul(void (*matmul_func)(const T *, const T *, T *, unsigned int, unsigned int),
                const char *func_name, unsigned int n, unsigned int block_dim,
                const T *h_a_shared, const T *h_b_shared) {
    size_t size = n * n;
    size_t bytes = size * sizeof(T);

    // Allocate host memory for result
    T *h_c;
    cudaMallocHost(&h_c, bytes);

    // Allocate device memory
    T *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);

    // Copy shared input matrices to device
    cudaMemcpy(d_a, h_a_shared, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b_shared, bytes, cudaMemcpyHostToDevice);

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call matmul function and measure time
    cudaEventRecord(start);
    matmul_func(d_a, d_b, d_c, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Results for " << func_name << ":\n";
    std::cout << "First element of C: " << h_c[0] << "\n";
    std::cout << "Last element of C: " << h_c[size - 1] << "\n";
    std::cout << "Time taken: " << ms << " ms\n";

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_c);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n block_dim\n";
        return 1;
    }

    // Parse arguments
    unsigned int n = std::atoi(argv[1]);
    unsigned int block_dim = std::atoi(argv[2]);
    size_t size = n * n;

    // Allocate shared input matrices
    float *h_a_float, *h_b_float;
    cudaMallocHost(&h_a_float, size * sizeof(float));
    cudaMallocHost(&h_b_float, size * sizeof(float));

    // Initialize shared matrices with random values
    initialize_matrix(h_a_float, size, -1.0f, 1.0f);
    initialize_matrix(h_b_float, size, -1.0f, 1.0f);

    // Convert float matrices to integer for matmul_1
    int *h_a_int = new int[size];
    int *h_b_int = new int[size];
    for (size_t i = 0; i < size; ++i) {
        h_a_int[i] = static_cast<int>(h_a_float[i]);
        h_b_int[i] = static_cast<int>(h_b_float[i]);
    }

    // Run tests for matmul_1, matmul_2, and matmul_3
    run_matmul<int>(matmul_1, "matmul_1 (int)", n, block_dim, h_a_int, h_b_int);
    run_matmul<float>(matmul_2, "matmul_2 (float)", n, block_dim, h_a_float, h_b_float);
    run_matmul<double>(matmul_3, "matmul_3 (double)", n, block_dim, h_a_float, h_b_float);

    // Free shared input matrices
    delete[] h_a_int;
    delete[] h_b_int;
    cudaFreeHost(h_a_float);
    cudaFreeHost(h_b_float);

    return 0;
}
