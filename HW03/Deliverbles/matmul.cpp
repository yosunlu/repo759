#include "matmul.h"
#include <vector>
#include <stdio.h>
#include <iostream>
using std::cout;
using std::vector;
using namespace std;

// This function produces a parallel version of matrix multiplication C = A B using OpenMP.
// The resulting C matrix should be stored in row-major representation.
void mmul(const float *A, const float *B, float *C, const std::size_t n)
{
#pragma omp parallel for
    // std::cout << "Inside a parallel block: " << omp_get_num_threads() << "\n";
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t k = 0; k < n; ++k)
        {
            for (size_t j = 0; j < n; ++j)
            {
                C[i * n + j] += A[i * n + k] * B[n * k + j];
                // cout << i * n + j << "\n";
            }
        }
    }
}

// int main()
// {
//     vector<float> A = {
//         1.0f, 2.0f,
//         3.0f, 4.0f};
//     vector<float> B = {
//         1.0f, 2.0f,
//         3.0f, 4.0f};
//     vector<float> C(4, 0.0f);
//     mmul(A.data(), B.data(), C.data(), 2);  // Pass the raw pointer using data()
//     for (int i = 0; i < 2; ++i)
//     {
//         for (int j = 0; j < 2; ++j)
//         {
//             cout << C[i * 2 + j] << "\n";
//         }
//     }
// }