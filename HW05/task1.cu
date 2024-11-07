#include <cuda.h>
#include <iostream>

__global__ void factorialKernal(){
    
    int factorial = 1;
    for(size_t i = 1; i <= threadIdx.x.x; ++i){
        factorial *= i; 
    }
    std::printf("%d!=%d\n", threadIdx.x.x, factorial);

}

int main(){

    factorialKernal<<<1, 8>>>();
    cudaDeviceSynchronize();

    return 1; 

}

