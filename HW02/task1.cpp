// The std::chrono namespace provides timer functions in C++
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
#include "scan.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>

// Provide some namespace shortcuts
using std::cout;
using std::vector;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

// Set some limits for the test
const size_t TEST_SIZE = 1000;
const size_t TEST_MAX = 32;

int main(int argc, char *argv[])
{

    // Ensure the program receives exactly one command-line argument
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    vector<float> vec(n, 0.0);

    // i) Creates an array of n random float numbers between -1.0 and 1.0
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for (int i = 0; i < n; i++)
    {
        vec[i] = dist(gen);
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // Get the starting timestamp
    start = high_resolution_clock::now();

    // ii) Scans the array using your scan function
    vector<float> output(n, 0.0);
    scan(vec.data(), output.data(), n);

    // Get the ending timestamp
    end = high_resolution_clock::now();

    // iii) Prints out the time taken by your scan function in milliseconds
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec.count() << "\n";



    // iv) Prints the first element of the output scanned array
    cout << output[0] << "\n";

    // v) Prints the last element of the output scanned array.
    cout << output[n - 1] << "\n";

    return 0;
}
