#include "matmul.h"
#include <stdio.h>
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>

// Provide some namespace shortcuts
using std::cout;
using std::vector;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using namespace std;

int main(int argc, char *argv[])
{
    size_t dim; // dimension of the 2 matrices
    size_t numThread; // number of threads
    // Should have at least 1 input; if 1 input is given, it is the number of threads; if 2 are given, the first in dim, the second in number of threads
    if (argc < 2)
    {
        std::cerr << "Usage: ./task2 n t, or ./task2 t, with n being the dimension of the 2 matrices, and t being the number of thread to execute.";
        return 1;
    }
    // parse the input
    else if (argc == 2)
    {
        dim = 10;
        numThread = std::atoi(argv[1]);
    }
    else
    {
        dim = std::atoi(argv[1]);
        numThread = std::atoi(argv[2]);
    }

    // variables to count time
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // variables to generate random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // initializing the 2 matrices
    vector<float> A(dim * dim, 0);
    vector<float> B(dim * dim, 0);
    float *C = new float[dim * dim]; // pointer to output array

    // fill in the matrics with random float
    for (size_t i = 0; i < dim; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
        {
            A[i * dim + j] = dist(gen);
            B[i * dim + j] = dist(gen);
        }
    }

    // Compute the matrix multiplication C = AB using  parallel implementation
    omp_set_num_threads(numThread);
#pragma omp parallel
#pragma omp master
    start = high_resolution_clock::now(); // Get the starting timestamp
    {
        mmul(A.data(), B.data(), C, dim); // Pass the raw pointer using data()
    }
    end = high_resolution_clock::now(); // Get the ending timestamp

    // Print the first element of the resulting C array.
    cout << C[0] << "\n";
    
    // Print the last element of the resulting C array.
    cout << C[dim * dim - 1] << "\n";

    // Print the time taken to run the mmul function in milliseconds.
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout <<duration_sec.count() << "\n"; // prints the amount of time taken in milliseconds

    // for (int i = 0; i < dim; ++i)
    // {
    //     for (int j = 0; j < dim; ++j)
    //     {
    //         cout << C[i * dim + j] << "\n";
    //     }
    // }
}