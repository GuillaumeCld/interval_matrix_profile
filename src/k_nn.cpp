#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "k_nn_lookup.hpp"
#include <algorithm>

using value_type = double;

int main(int argc, char const *argv[])
{
    std::vector<value_type> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1000); // Random numbers between 0 and 1

    const int size = 1000000;
    const int k = 5;
    const int exclude_zone = 15;

    for (int i = 0; i < size; ++i)
    {
        data.push_back(dis(gen));
        // data.push_back(i);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto result = k_linear_sweeps(data, k, exclude_zone);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

    for (int i = 0; i < k; ++i)
    {
        std::cout << "Min value: " << result.first[i] << " at index: " << result.second[i] << std::endl;
    }

    start = std::chrono::high_resolution_clock::now();
    result = k_nn_heap(data, k, exclude_zone);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

    for (int i = 0; i < k; ++i)
    {
        std::cout << "Min value: " << result.first[i] << " at index: " << result.second[i] << std::endl;
    }

    return 0;
}
