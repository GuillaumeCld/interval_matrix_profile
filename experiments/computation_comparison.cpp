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

using value_type = double;

void testMatrixProfileComputationSpeed(int vector_size, int window_size)
{
    // Generate random vector

    std::vector<value_type> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    for (int i = 0; i < vector_size; ++i)
    {
        data.push_back(dis(gen));
    }
    // Measure computation time
    // auto start_time = std::chrono::high_resolution_clock::now();
    // auto mpOutput = computeMatrixProfileBruteForce(data, window_size);
    // auto matrix_profile = std::get<0>(mpOutput);
    // auto end_time = std::chrono::high_resolution_clock::now();

    // // // Compute duration
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    // // // Output results
    // std::cout << "Brute Force " << vector_size << " " << window_size;
    // std::cout << " " << duration.count() << std::endl;

    // Measure computation time
    const int exclude = window_size / 2;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mpOutput = blockSTOMP_v2(data, window_size, 5000, 5000, exclude);
    auto matrix_profile_blockstomp = std::get<0>(mpOutput);
    auto end_time = std::chrono::high_resolution_clock::now();
    // Compute duration
    auto duration =  std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "STOMP " << vector_size << " " << window_size;
    std::cout << " " << duration.count() << std::endl;
}

void compute_bf_n( int window_size)
{
    // Generate random vector
    std::vector<int> sizes = {2<<16, 2<<17, 2<<18, 2<<19, 2<<20};
    const int exclude = window_size / 2;

    for (auto const &vector_size: sizes ){
        std::vector<value_type> data;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

        for (int i = 0; i < vector_size; ++i)
        {
            data.push_back(dis(gen));
        }
        // Measure computation time
        auto start_time = std::chrono::high_resolution_clock::now();
        auto mpOutput = computeMatrixProfileBruteForce(data, window_size, exclude);
        auto matrix_profile = std::get<0>(mpOutput);
        auto end_time = std::chrono::high_resolution_clock::now();

        // // Compute duration
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        // // Output results
        std::cout << "Brute Force " << vector_size << " " << window_size;
        std::cout << " " << duration.count() << std::endl;
    }
}
void compute_stomp_n( int window_size)
{
    // Generate random vector
    std::vector<int> sizes = {2<<16, 2<<17, 2<<18, 2<<19, 2<<20};
    for (auto const &vector_size: sizes ){
        std::vector<value_type> data;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

        for (int i = 0; i < vector_size; ++i)
        {
            data.push_back(dis(gen));
        }
        const int exclude = window_size / 2;

        // Measure computation time
        std::cout << "Starting " << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto mpOutput = blockSTOMP_v2(data, window_size, 10000, 10000, exclude);
        auto matrix_profile = std::get<0>(mpOutput);
        auto end_time = std::chrono::high_resolution_clock::now();

        // // Compute duration
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        // // Output results
        std::cout << "STOMP " << vector_size << " " << window_size;
        std::cout << " " << duration.count() << std::endl;
    }
}
void block_dims(int vector_size, int window_size)
{
    // Generate random vector
    std::vector<value_type> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    for (int i = 0; i < vector_size; ++i)
    {
        data.push_back(dis(gen));
    }

    // Open file to write results
    std::ofstream outfile("../experiments/computation_times.txt");

    if (!outfile.is_open())
    {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }
    const int exclude = window_size / 2;

    // Loop over different block widths and heights
    for (int block_height = 500; block_height <= 5000; block_height += 500)
    {
        for (int block_width = 500; block_width <= 5000; block_width += 500)
        {
            // std::cout << "Block Width: " << block_width << ", Block Height: " << block_height << std::endl;
            auto start_time = std::chrono::high_resolution_clock::now();
            auto mpOutput = blockSTOMP_v2(data, window_size, block_width, block_height, exclude);
            auto end_time = std::chrono::high_resolution_clock::now();

            // Compute duration
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            // Write results to file
            outfile << block_width << "," << block_height << ","
                    << vector_size << "," << window_size << ","
                    << duration.count() << std::endl;

            // Optionally print to console
            std::cout << "Block Width: " << block_width << ", Block Height: " << block_height
                      << ", Vector Size: " << vector_size << ", Window Size: " << window_size
                      << ", Duration: " << duration.count() << " ms" << std::endl;
        }
    }

    // Close the file
    outfile.close();
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: ./program_name vector_size window_size" << std::endl;
        return 1;
    }

    int vector_size = std::stoi(argv[1]);
    int window_size = std::stoi(argv[2]);

    // testMatrixProfileComputationSpeed(vector_size, window_size);
    // block_dims(vector_size, window_size);
    compute_stomp_n(window_size);   
    // compute_bf_n(window_size);

    return 0;
}