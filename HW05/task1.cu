#include <cuda.h>
#include <stdio.h>

__global__ void factorialKernal(){
    
    int factorial = 1;
    for(size_t i = 1; i <= threadIdx.x; ++i){
        factorial *= i; 
    }
    printf("%d!=%d\n", threadIdx.x, factorial);

}

int main(){

    factorialKernal<<<1, 8>>>();
    cudaDeviceSynchronize();
    
    return 0; 

}

