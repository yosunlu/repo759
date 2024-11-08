#include <cuda.h>
#include <stdio.h>

__global__ void factorialKernel(){
    
    int factorial = 1;
    for(size_t i = 1; i <= threadIdx.x + 1 ; ++i){
        factorial *= i; 
    }
    printf("%d!=%d\n", threadIdx.x + 1, factorial);

}

int main(){

    factorialKernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    
    return 0; 

}

