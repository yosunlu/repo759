#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include "convolution.h"
#define M 3

// Provide some namespace shortcuts
using std::cout;
using std::vector;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[])
{
    // Ensure the program receives exactly one command-line argument
    if (argc != 3)
    {
        std::cerr << "Usage: ./task2 n m";
        return 1;
    }

    // parse the input
    int dim = std::atoi(argv[1]);
    int numThread = std::atoi(argv[2]);

    // variables to count time
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // i) Creates an dim×dim image matrix (stored in 1D in row-major order) of random float numbers between -10.0 and 10.0
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0, 10.0);
    vector<float> image(dim * dim, 0.0f);

    for (size_t i = 0; i < dim; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
        {
            image[i * dim + j] = dist(gen);
        }
    }

    // ii) Creates an 3x3 mask matrix (stored in 1D in row-major order) of random float numbers between -1.0 and 1.0
    std::uniform_real_distribution<float> dist_1(-1.0, 1.0);
    vector<float> mask(M * M, 0.0f);

    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < M; ++j)
        {
            mask[i * M + j] = dist_1(gen);
        }
    }

    // Get the starting timestamp
    start = high_resolution_clock::now();

    // iv) Apply the mask matrix to the image using your convolve function with t threads
    vector<float> output(dim * dim, 0.0f);
    omp_set_num_threads(numThread);
    convolve(image.data(), output.data(), dim, mask.data(), M);

    // Get the ending timestamp
    end = high_resolution_clock::now();

    // v) Prints the first element of the resulting convolved array
    cout << output[0] << "\n";

    // vi) Prints the last element of the resulting convolved array
    cout << output[dim * dim - 1] << "\n";

    // vii) Prints out the time taken by your convolve function in milliseconds.
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec.count() << "\n";
}