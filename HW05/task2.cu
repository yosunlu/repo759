#include <cuda.h>
#include <stdio.h>
#include <random>

__global__ void kernel(int* dA, int a){
    
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    dA[index] = a * threadIdx.x + blockIdx.x;
}

int main(){

    // From the host, allocates an array of 16 ints on the device called dA.
    const int numElems = 16;
    int *dA;
    cudaMalloc((void**)&dA, numElems * sizeof(int));
    cudaMemset(dA, 0, numElems * sizeof(int));

    // generate random number a between -100 and 100
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-100, 100);
    const int a = dist(gen);

    // Launches a kernel with 2 blocks, each block having 8 threads.
    const int numBlocks = 2;
    const int numThreads = 8;
    kernel<<<numBlocks, numThreads>>>(dA, a); 
    cudaDeviceSynchronize();

    // Copies back the data stored in the device array dA into a host array called hA
    int hostArray[numElems];
    cudaMemcpy(&hostArray, dA, sizeof(int) * numElems, cudaMemcpyDeviceToHost);


    // Prints (from the host) the 16 values stored in the host array separated by a single space each.
    for(int i = 0; i < numElems; ++i){
        printf("%d ", hostArray[i]); 
    }
    
    // Free device memory
    cudaFree(dA);

    return 0; 

}

