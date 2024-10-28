#include "msort.h"
#include <iostream>
#include <algorithm> // for std::sort
#include <omp.h>
#include <iostream>
#include <vector>

// Function to merge two sorted halves of an array
void merge(int *arr, int *temp, std::size_t left, std::size_t mid, std::size_t right)
{
    std::size_t i = left;    // Starting index for left subarray
    std::size_t j = mid + 1; // Starting index for right subarray
    std::size_t k = left;    // Starting index to be sorted

    // Merge the two subarrays
    while (i <= mid && j <= right)
    {
        if (arr[i] <= arr[j])
        {
            temp[k++] = arr[i++];
        }
        else
        {
            temp[k++] = arr[j++];
        }
    }

    // Copy the remaining elements of left subarray (if any)
    while (i <= mid)
    {
        temp[k++] = arr[i++];
    }

    // Copy the remaining elements of right subarray (if any)
    while (j <= right)
    {
        temp[k++] = arr[j++];
    }

    // Copy the sorted elements back to the original array
    for (i = left; i <= right; i++)
    {
        arr[i] = temp[i];
    }
}

// Recursive parallel merge sort function
void parallel_merge_sort(int *arr, int *temp, std::size_t left, std::size_t right, std::size_t threshold)
{
    // Base case: if the size of the array is below or equal to the threshold, use serial sort
    if (right - left + 1 <= threshold)
    {
        std::sort(arr + left, arr + right + 1); // Use std::sort for small arrays
        return;
    }

    if (left < right)
    {
        // Find the middle point
        std::size_t mid = left + (right - left) / 2;

        // Create parallel tasks for sorting the two halves

#pragma omp task // Task for the left half
        parallel_merge_sort(arr, temp, left, mid, threshold);
// #pragma omp task // Task for the left half
        parallel_merge_sort(arr, temp, mid + 1, right, threshold);

        #pragma omp taskwait // Ensure both tasks are finished before merging

        // Merge the sorted halves
        merge(arr, temp, left, mid, right);
    }
}

// This function does a merge sort on the input array "arr" of length n
// "threshold" is the lower limit of array size where your function would
// start making parallel recursive calls. If the size of array goes below
// the threshold, a serial sort algorithm will be used to avoid overhead
// of task scheduling

void msort(int *arr, const std::size_t n, const std::size_t threshold)
{
    // Temporary array for merging
    int *temp = new int[n];

#pragma omp parallel
    {
#pragma omp single
        // Start the parallel merge sort
        parallel_merge_sort(arr, temp, 0, n - 1, threshold);
    }

    // Free the temporary array
    delete[] temp;
}

// int main(){
//     std::vector<int> arr = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
//     msort(arr.data(), arr.size(), 1);
//     for(int i = 0; i < 10; i++){
//         std::cout << arr[i] << " ";
//     }
// }