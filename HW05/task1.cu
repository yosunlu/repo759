#include <cuda.h>
#include <iostream>

__global__ void factorialKernal(){
    
    int factorial = 1;
    for(int i = 1; i <= threadId.x; ++i){
        factorial *= i; 
    }
    std::prinf("%d!=%d\n", threadId.x, factorial);

}

int main(){

    factorialKernal<<<1, 8>>>();
    cudaDeviceSynchronize();

    return 1; 

}

