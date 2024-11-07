#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n){

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    for(int i = 0; i < n; ++i){
        b[index] *= a[index];
    }
}