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

auto compute(std::string folder)
{
    std::vector<double> data;
    std::cout << "Reading data" << std::endl;
    std::string path = "../Data/climate_pulse/";
    std::string filename = path + folder + "/daily_series.txt";
    readFile<double>(filename.c_str(), data, "%lf");
    // Define the window size, exclude value, and seasons
    const int window_size = 7;
    const int exclude = 4;
    const int k = 3;
    const int interval_length = 90;
    const int n = data.size() - window_size + 1;
    std::cout << "Data size: " << data.size() << ", window size: " << window_size << ", n: " << n << ", Exclude: " << exclude << ", k: " << k << std::endl;

    std::vector<int> period_starts;
    std::string period_starts_filename = path + folder + "/periods_start_daily.txt";
    readFile<int>(period_starts_filename.c_str(), period_starts, "%d");

    std::cout << "Starting computation" << std::endl;
    std::cout << "Matrix Profile" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = computeMatrixProfileBruteForce(data, window_size, exclude);
    std::vector<double> mp = std::get<0>(result);
    std::vector<int> mp_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "MP BF execution time: " << duration.count() << " ms" << std::endl;

    std::cout << "Interval Matrix Profile" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp = std::get<0>(result);
    std::vector<int> imp_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "IMP BF execution time: " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_STOMP_kNN(data, window_size, period_starts, interval_length, exclude, k);
    std::vector<double> imp_3NN = std::get<0>(result);
    std::vector<int> imp_3NN_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "IMP STOMP kNN execution time: " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_STOMP_kNN(data, window_size, period_starts, interval_length, exclude, 5);
    std::vector<double> imp_5NN = std::get<0>(result);
    std::vector<int> imp_5NN_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "IMP STOMP kNN execution time: " << duration.count() << " ms" << std::endl;


    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_STOMP_kNN(data, window_size, period_starts, interval_length, exclude, 10);
    std::vector<double> imp_10NN = std::get<0>(result);
    std::vector<int> imp_10NN_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "IMP STOMP kNN execution time: " << duration.count() << " ms" << std::endl;

    std::string mp_filename = path + folder + "/mp.txt";
    std::string imp_filename = path + folder + "/imp.txt";
    std::string imp_3NN_filename = path + folder + "/imp_3NN.txt";
    std::string imp_5NN_filename = path + folder + "/imp_5NN.txt";
    std::string imp_10NN_filename = path + folder + "/imp_10NN.txt";
    write_vector_to_file(mp_filename.c_str(), mp);
    write_vector_to_file(imp_filename.c_str(), imp);
    write_vector_to_file(imp_3NN_filename.c_str(), imp_3NN);
    write_vector_to_file(imp_5NN_filename.c_str(), imp_5NN);
    write_vector_to_file(imp_10NN_filename.c_str(), imp_10NN);

    std::string mp_index_filename = path + folder + "/mp_index.txt";
    std::string imp_index_filename = path + folder + "/imp_index.txt";
    std::string imp_3NN_index_filename = path + folder + "/imp_3NN_index.txt";
    std::string imp_5NN_index_filename = path + folder + "/imp_5NN_index.txt";
    std::string imp_10NN_index_filename = path + folder + "/imp_10NN_index.txt";
    write_vector_to_file(mp_index_filename.c_str(), mp_index);
    write_vector_to_file(imp_index_filename.c_str(), imp_index);
    write_vector_to_file(imp_3NN_index_filename.c_str(), imp_3NN_index);
    write_vector_to_file(imp_5NN_index_filename.c_str(), imp_5NN_index);
    write_vector_to_file(imp_10NN_index_filename.c_str(), imp_10NN_index);
}




int main()
{
  compute("SST");
  compute("T2M");

  return 0;
}
