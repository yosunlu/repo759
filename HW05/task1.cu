#include <cuda.h>
#include <stdio.h>

__global__ void factorialKernel(){
    
    int factorial = 1;
    for(size_t i = 1; i <= threadIdx.x; ++i){
        factorial *= i; 
    }
    printf("%d!=%d\n", threadIdx.x, factorial);

}

int main(){

    printf("test begin");
    factorialKernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    printf("test end");
    
    return 0; 

}

