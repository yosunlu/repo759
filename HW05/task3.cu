#include "vscale.cuh"
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>

// Provide some namespace shortcuts
using std::cout;
using std::vector;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[]){

    if (argc != 2)
    {
        std::cerr << "Usage: ./task3 n, where n is the length of the array";
        return 1;
    }


    // parse the arguments
    size_t n = std::atoi(argv[1]);

    // Creates two arrays of length n filled by random numbers1 where n is read from the first command line argument. 
    // The range of values for array a is [-10.0, 10.0], 
    // whereas the range of values for array b is [0.0, 1.0].
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<int> dist(-10.0, 10.0);
    std::uniform_real_distribution<int> dist1(0.0, 1.0);

    float* a;
    float* b;
    for(int i = 0; i < n; ++i){
        a[i] = dist(gen);
        b[i] = dist1(gen);
    }

    // Calls your vscale kernel with a 1D execution configuration that uses 512 threads per block.
    int numThreadsPerBlock = 512; 
    // ensures there are enough blocks so there are enough threads to cover the length of the array.
    int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock; 

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Calls the kernel
    vscale<<<numBlocks, numThreadsPerBlock>>>(a, b, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Prints the amount of time taken to execute the kernel in milliseconds using CUDA events
    printf("%f\n", ms);

    //  Prints the first element of the resulting array.
    printf("%f\n,", b[0]);

    // Prints the last element of the resulting array.
    printf("%f\n,", b[n-1]);

}