#include <iostream>
#include <vector>
#include <chrono> // For timing
#include <random> // For random number generation
#include <algorithm>
#include <cmath>
#include <span>
#include <fstream>
#include <matrix_profile.hpp>
#include <seasonal_matrix_profile.hpp>
#include <interval_matrix_profile.hpp>
#include <utils.hpp>

using value_type= double;


void testMatrixProfileComputationSpeed(int vector_size, int window_size) {
    // Generate random vector

    std::vector<value_type> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    for (int i = 0; i < vector_size; ++i) {
        data.push_back(dis(gen));
    }
    // Measure computation time
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mpOutput = computeMatrixProfileBruteForce(data, window_size);
    auto matrix_profile = std::get<0>(mpOutput);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // // Compute duration
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    // // Output results
    std::cout << "Brute Force " << vector_size << " " << window_size;
    std::cout << " " << duration.count() << std::endl;

    // Measure computation time
    // auto start_time = std::chrono::high_resolution_clock::now();
    // auto mpOutput = blockSTOMP_v2(data, window_size, 5000, 5000);
    // auto matrix_profile_blockstomp = std::get<0>(mpOutput);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // // Compute duration
    // auto duration =  std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    // std::cout << "STOMP " << vector_size << " " << window_size;
    // std::cout << " " << duration.count() << std::endl;

}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./program_name vector_size window_size" << std::endl;
        return 1;
    }
    
    int vector_size = std::stoi(argv[1]);
    int window_size = std::stoi(argv[2]);
    
    testMatrixProfileComputationSpeed(vector_size, window_size);
    
    return 0;
}