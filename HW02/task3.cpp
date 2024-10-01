// The std::chrono namespace provides timer functions in C++
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include "matmul.h"
#define N 10

// Provide some namespace shortcuts
using std::cout;
using std::vector;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using namespace std;

int main(int argc, char *argv[])
{
    // variables to count time
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // generates square matrices A and B of dimension at least 1000×1000 stored in row-major order.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    cout << N << "\n";
    vector<double> A(N * N, 0);
    vector<double> B(N * N, 0);
    double *C = new double[N * N]; // pointer to output array

    // fill in the C array with random doubles
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            A[i * N + j] = dist(gen);
            B[i * N + j] = dist(gen);
        }
    }

    // computes the matrix product C = AB using each of your functions

    start = high_resolution_clock::now(); // Get the starting timestamp
    mmul1(A.data(), B.data(), C, N);      // the matrix multiplication method 1
    end = high_resolution_clock::now();   // Get the ending timestamp
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec.count() << "\n"; // prints the amount of time taken in milliseconds
    cout << C[N * N - 1] << "\n";         // prints the last element of the resulting C

    fill(C, C + (N * N), 0.0); // reset C to zeros
    start = high_resolution_clock::now(); 
    mmul2(A.data(), B.data(), C, N); // the matrix multiplication method 2
    end = high_resolution_clock::now(); 
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec.count() << "\n"; 
    cout << C[N * N - 1] << "\n";
    
    fill(C, C + (N * N), 0.0);
    start = high_resolution_clock::now(); 
    mmul3(A.data(), B.data(), C, N); // the matrix multiplication method 3
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec.count() << "\n"; 
    cout << C[N * N - 1] << "\n";

    fill(C, C + (N * N), 0.0);
    start = high_resolution_clock::now(); 
    mmul4(A, B, C, N); // the matrix multiplication method 4
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec.count() << "\n"; 
    cout << C[N * N - 1] << "\n";

    delete[] C;

    return 0;
}