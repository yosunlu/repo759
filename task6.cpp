#include <iostream>
#include <cstdlib>  
#include <cstdio>

int main(int argc, char* argv[]) {
    // Ensure the program receives exactly one command-line argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);

    // a) Print integers from 0 to N in ascending order 
    for (int i = 0; i <= N; ++i) {
        std::printf("%d ", i);
    }
    std::printf("\n");

    // b) Print integers from N to 0 in descending order
    for (int i = N; i >= 0; --i) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    return 0;
}
