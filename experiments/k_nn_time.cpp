#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstdlib>
#include "k_nn_lookup.hpp"

using value_type = double;

int main(int argc, char const *argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <size> <k> <exclude_zone>" << std::endl;
        return 1;
    }
    const int size = std::atoi(argv[1]);
    const int k = std::atoi(argv[2]);
    const int exclude_zone = std::atoi(argv[3]);

    std::vector<value_type> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1); 


    for (int i = 0; i < size; ++i)
    {
        data.push_back(dis(gen));
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto result = k_linear_sweeps(data, k, exclude_zone);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =  std::chrono::duration_cast<std::chrono::milliseconds>(end- start);
    std::cout << "Sweep " << elapsed_seconds.count() <<"\n";

    start = std::chrono::high_resolution_clock::now();
    result = k_nn_heap(data, k, exclude_zone);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds =  std::chrono::duration_cast<std::chrono::milliseconds>(end- start);
    std::cout << "Heap " << elapsed_seconds.count() << "\n";

    return 0;
}
