#include <cuda.h>
#include <iostream>

__global__ void factorialKernal(){
    
    int factorial = 1;
    for(size_t i = 1; i <= threadIdx.x; ++i){
        factorial *= i; 
    }
    std::printf("%d!=%d\n", threadIdx.x, factorial);

}

int main(){

    factorialKernal<<<1, 8>>>();
    cudaDeviceSynchronize();
    std::cout << "test" << endl;

    return 1; 

}

