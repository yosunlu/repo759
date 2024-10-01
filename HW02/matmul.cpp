#include "matmul.h"
#include <iostream>
using namespace std;

// mmul1 should have three for loops: the outer loop sweeps index i through the rows of C, the middle loop sweeps index j through the columns of C,
// and the innermost loop sweeps index k through; i.e., to carry out, the dot product of the ith row A with the jth column of B.
// Inside the innermost loop, you should have a single line of code which increments Cij.
// Assume that A and B are 1D arrays storing the matrices in row-major order.
void mmul1(const double *A, const double *B, double *C, const unsigned int n)
{
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            for (size_t k = 0; k < n; ++k)
            {
                C[i * n + j] += A[i * n + k] * B[n * k + j];
                // cout << i * n + j << "\n";
            }
        }
    }
}
// mmul2 should also have three for loops, but the two innermost loops should be swapped relative to mmul1
// (such that, if your original iterators are from outer to inner (i,j,k), then they now become (i,k,j)).
// That is the only difference between mmul1 and mmul2.
void mmul2(const double *A, const double *B, double *C, const unsigned int n)
{

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
// mmul3 should also have three for loops, but the outermost loop in mmul1 should become the innermost loop in mmul3,
// and the other 2 loops do not change their relative positions (such that, if your original iterators are from outer to inner (i,j,k), then they now become (j,k,i)).
// That is the only difference between mmul1 and mmul3.
void mmul3(const double *A, const double *B, double *C, const unsigned int n)
{
    for (size_t j = 0; j < n; ++j)
    {
        for (size_t k = 0; k < n; ++k)
        {
            for (size_t i = 0; i < n; ++i)
            {
                C[i * n + j] += A[i * n + k] * B[n * k + j];
                // cout << i * n + j << "\n";
            }
        }
    }
}

//  mmul4 should have the for loops ordered as in mmul1, but this time around A and B are stored as std::vector<double>. 
// That is the only difference between mmul1 and mmul4.
void mmul4(const std::vector<double> &A, const std::vector<double> &B, double *C, const unsigned int n)
{
    {
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                for (size_t k = 0; k < n; ++k)
                {
                    C[i * n + j] += A[i * n + k] * B[n * k + j];
                    // cout << i * n + j << "\n";
                }
            }
        }
    }
}

// int main()
// {
//     vector<double> A = {
//         1, 2,
//         3, 4};
//     vector<double> B = {
//         1, 2,
//         3, 4};
//     vector<double> C(4, 0);
//     mmul4(A, B, C.data(), 2);
//     for (int i = 0; i < 2; ++i)
//     {
//         for (int j = 0; j < 2; ++j)
//         {
//             cout << C[i * 2 + j] << "\n";
//         }
//     }
// }