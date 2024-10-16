#include "convolution.h"
#include "vector"
#include "iostream"
using namespace std;

// Function to handle padding based on the given conditions
int paddedValue(int i, int j, int n)
{
    if ((i >= 0 && i < n) || (j >= 0 && j < n))
    {
        return 1; // Edges (excluding corners)
    }
    return 0; // Corners and outside the edges
}

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    int half_m = (m - 1) / 2;
#pragma omp parallel for
// iterate the image rows
    for (size_t i = 0; i < n; ++i)
    {
        // iterate the image columns
        for (size_t j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            // iterate the mask rows
            for (size_t r = 0; r < m; ++r)
            {
                // iterate the mask columns
                for (size_t c = 0; c < m; ++c)
                {
                    size_t imgX = i + r - half_m;
                    size_t imgY = j + c - half_m;
                    float fValue = (imgX >= 0 && imgX < n && imgY >= 0 && imgY < n) ? image[imgX * n + imgY] : paddedValue(imgX, imgY, n);
                    sum += mask[r * m + c] * fValue;
                }
            }
            output[i * n + j] = sum;
        }
    }
}

// int main() {
//     // Example image (n x n)
//     vector<float> image = {
//         1, 3, 4, 8,
//         6, 5, 2, 4,
//         3, 4, 6, 8,
//         1, 4, 5, 2
//     };

//     // Example mask (m x m)
//     vector<float> mask = {
//         0, 0, 1,
//         0, 1, 0,
//         1, 0, 0
//     };

//     // Perform convolution
//     size_t n = 4;
//     size_t m = 3;
//     vector<float> output(n * n, 0.0f);

//     convolve(image.data(), output.data(), n, mask.data(), m);

//     // Display result
//     for (size_t i = 0; i < n; ++i) {
//         for (size_t j = 0; j < n; ++j) {
//             cout << output[i * n + j] << " ";
//         }
//         cout << endl;
//     }
//     return 0;
// }