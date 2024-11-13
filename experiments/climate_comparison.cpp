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
#include <bimp.hpp>
#include <bimb_knn.hpp>
#include <utils.hpp>

using value_type = double;

auto daily_climate_series()
{
    std::vector<double> data;
    std::cout << "Reading data" << std::endl;
    readFile<double>("../Data/ERA5_land/daily_series.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    const int window_size = 7;
    const int exclude = 2;
    const int k = 3;
    const int interval_length = 90;
    const int n = data.size() - window_size + 1;
    std::cout << "Data size: " << data.size() << ", window size: " << window_size << ", n: " << n << ", Exclude: " << exclude << ", k: " << k << std::endl;

    std::cout << "Building seasons" << std::endl;
    std::vector<std::vector<std::pair<int, int>>> seasons;
    seasons.push_back(build_season_vector<double>("../Data/ERA5_land/seasons_daily_0.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/ERA5_land/seasons_daily_1.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/ERA5_land/seasons_daily_2.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/ERA5_land/seasons_daily_3.txt", n));
    std::cout << "Reading period starts" << std::endl;
    std::vector<int> period_starts;
    readFile<int>("../Data/ERA5_land/periods_start_daily.txt", period_starts, "%d");

    std::cout << "Starting computation" << std::endl;
    std::cout << "Matrix Profile" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = computeMatrixProfileBruteForce(data, window_size, exclude);
    std::vector<double> mp_bf = std::get<0>(result);
    std::vector<int> mp_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "MP BF execution time: " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = blockSTOMP_v2(data, window_size, 5000, 5000, exclude);
    std::vector<double> mp_stomp = std::get<0>(result);
    std::vector<int> mp_stomp_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "MP STOMP execution time: " << duration.count() << " ms" << std::endl;


    std::cout << "Seasonal Matrix Profile" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    result = seasonal_matrix_profile_brute_force(data, window_size, exclude, seasons);
    std::vector<double> smp_bf = std::get<0>(result);
    std::vector<int> smp_bf_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "SMP BF execution time: " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = seasonal_matrix_profile_STOMP_blocking(data, window_size, exclude, seasons);
    std::vector<double> smp_stomp = std::get<0>(result);
    std::vector<int> smp_stomp_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "SMP STOMP execution time: " << duration.count() << " ms" << std::endl;


    std::cout << "Interval Matrix Profile" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result);
    std::vector<int> imp_bf_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "IMP BF execution time: " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = BIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result);
    std::vector<int> imp_stomp_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "IMP STOMP execution time: " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_STOMP_kNN(data, window_size, period_starts, interval_length, exclude, k);
    std::vector<double> imp_stomp_kNN = std::get<0>(result);
    std::vector<int> imp_stomp_kNN_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "IMP STOMP kNN execution time: " << duration.count() << " ms" << std::endl;

    write_vector_to_file("../Data/ERA5_land/daily_outputs/mp_bf.txt", mp_bf);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/mp_stomp.txt", mp_stomp);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/smp_bf.txt", smp_bf);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/smp_stomp.txt", smp_stomp);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/imp_bf.txt", imp_bf);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/imp_stomp.txt", imp_stomp);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/imp_stomp_kNN.txt", imp_stomp_kNN);

    write_vector_to_file("../Data/ERA5_land/daily_outputs/mp_bf_index.txt", mp_index);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/mp_stomp_index.txt", mp_stomp_index);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/smp_bf_index.txt", smp_bf_index);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/smp_stomp_index.txt", smp_stomp_index);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/imp_bf_index.txt", imp_bf_index);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/imp_stomp_index.txt", imp_stomp_index);
    write_vector_to_file("../Data/ERA5_land/daily_outputs/imp_stomp_kNN_index.txt", imp_stomp_kNN_index); 
}


auto hourly_climate_series()
{
    std::vector<double> data;
    std::cout << "Reading data" << std::endl;
    readFile<double>("../Data/ERA5_land/hourly_series.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    const int window_size = 7 * 24;
    const int exclude = 4 * 24;
    const int k = 3;
    const int n = data.size() - window_size + 1;
    std::cout << "Data size: " << data.size() << ", window size: " << window_size << ", n: " << n << ", Exclude: " << exclude << ", k: " << k << std::endl;
    std::cout << "Building seasons" << std::endl;
    std::vector<std::vector<std::pair<int, int>>> seasons;
    seasons.push_back(build_season_vector<double>("../Data/ERA5_land/seasons_hourly_0.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/ERA5_land/seasons_hourly_1.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/ERA5_land/seasons_hourly_2.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/ERA5_land/seasons_hourly_3.txt", n));
    const int interval_length = 90 * 24;
    std::cout << "Reading period starts" << std::endl;
    std::vector<int> period_starts;
    readFile<int>("../Data/ERA5_land/periods_start_hourly.txt", period_starts, "%d");
    std::cout << "Starting computation" << std::endl;
    std::cout << "Matrix Profile" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = blockSTOMP_v2(data, window_size, 5000, 5000, exclude);
    std::vector<double> mp_stomp = std::get<0>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "MP STOMP execution time: " << duration.count() << " s" << std::endl;


    std::cout << "Seasonal Matrix Profile" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    result = seasonal_matrix_profile_STOMP_blocking(data, window_size, exclude, seasons);
    std::vector<double> smp_stomp = std::get<0>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "SMP STOMP execution time: " << duration.count() << " s" << std::endl;


    std::cout << "Interval Matrix Profile" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = BIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "IMP STOMP execution time: " << duration.count() << " s" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_STOMP_kNN(data, window_size, period_starts, interval_length, exclude, k);
    std::vector<double> imp_stomp_kNN = std::get<0>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "IMP STOMP kNN execution time: " << duration.count() << " s" << std::endl;

    write_vector_to_file("../Data/ERA5_land/hourly_outputs/mp_stomp.txt", mp_stomp);
    write_vector_to_file("../Data/ERA5_land/hourly_outputs/smp_stomp.txt", smp_stomp);
    write_vector_to_file("../Data/ERA5_land/hourly_outputs/imp_stomp.txt", imp_stomp);
    write_vector_to_file("../Data/ERA5_land/hourly_outputs/imp_stomp_kNN.txt", imp_stomp_kNN);
}



int main()
{
  daily_climate_series();
  // hourly_climate_series();

  return 0;
}
