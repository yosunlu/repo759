// The std::chrono namespace provides timer functions in C++
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include "convolution.h"

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
    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);

    // variables to count time
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // i) Creates an n×n image matrix (stored in 1D in row-major order) of random float numbers between -10.0 and 10.0
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0, 10.0);
    vector<float> image(n * n, 0.0f);

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            image[i * n + j] = dist(gen);
        }
    }

    // ii) Creates an m×m mask matrix (stored in 1D in row-major order) of random float numbers between -1.0 and 1.0
    std::uniform_real_distribution<float> dist_1(-1.0, 1.0);
    vector<float> mask(m * m, 0.0f);

    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            mask[i * m + j] = dist_1(gen);
        }
    }

    // iii) Applies the mask to image using your convolve function
    vector<float> output(n * n, 0.0f);
    // Get the starting timestamp
    start = high_resolution_clock::now();
    convolve(image.data(), output.data(), n, mask.data(), m);
    // Get the ending timestamp
    end = high_resolution_clock::now();

    // iv) Prints out the time taken by your convolve function in milliseconds.
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    // Convert the calculated duration to a double using the standard library
    
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec.count() << "\n";

    // v) Prints the first element of the resulting convolved array
    cout << output[0] << "\n";

    // vi) Prints the last element of the resulting convolved array
    cout << output[n * n - 1] << "\n";

    return 0;
}