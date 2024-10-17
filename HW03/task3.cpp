#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <omp.h>
#include "msort.h"

// Provide some namespace shortcuts
using std::cout;
using std::vector;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[])
{
    // Ensure the program receives exactly 3 command-line argument
    // n is a positive integer for array length, t is an integer in the range [1, 20], ts is the
    // threshold as the lower limit to make recursive calls in order to avoid the overhead of recursion/task scheduling when the input array has small size;
    // under this limit, a serial sorting algorithm without recursion calls will be used

    if (argc != 4)
    {
        std::cerr << "Usage: ./task2 n t ts";
        return 1;
    }

    // variables to count time
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // parse the arguments
    size_t n = std::atoi(argv[1]);
    size_t t = std::atoi(argv[2]);
    size_t ts = std::atoi(argv[3]);

    // Create array arr and fill with random int type numbers in the range [-1000, 1000]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000, 1000);
    vector<int> arr(n, 0);

    for (size_t i = 0; i < n; ++i)
    {
        arr[i] = dist(gen);
    }

    // Get the starting timestamp
    start = high_resolution_clock::now();
    // Set the number of threads for OpenMP
    omp_set_num_threads(t);
    // Apply your msort function to the arr. Set number of threads to t, which is the second command line argument
    msort(arr.data(), n, ts);

    // Get the ending timestamp
    end = high_resolution_clock::now();

    // Print the first element of the resulting arr array
    cout << arr[0] << "\n";

    // Print the last element of the resulting arr array
    cout << arr[n - 1] << "\n";

    // Print the time taken to run the msort function in milliseconds

    // for(int i = 0; i < n; i++){
    //     cout << arr[i] << "\n";
    // }

    // vi) Prints out the time taken by your convolve function in milliseconds.
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout <<  duration_sec.count() << "\n";
    return 1;
}
