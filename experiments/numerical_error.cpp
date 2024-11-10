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
#include <complex>

using value_type = double;

auto compute_cp(std::string folder)
{
    std::vector<float> data;
    std::cout << "Reading data" << std::endl;
    std::string path = "../Data/climate_pulse/";
    std::string filename = path + folder + "/daily_series.txt";
    readFile<float>(filename.c_str(), data, "%f");
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

    std::vector<double> double_data = std::vector<double>(data.begin(), data.end());
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = interval_matrix_profile_brute_force(double_data, window_size, period_starts, interval_length, exclude);
    std::vector<double> double_imp = std::get<0>(result);
    std::vector<float> imp = std::vector<float>(double_imp.begin(), double_imp.end());
    std::vector<int> imp_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BF " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    auto result1 = modified_AAMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<float> imp_aamp = std::get<0>(result1);
    std::vector<int> imp_aamp_index = std::get<1>(result1);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "aamp " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    auto result2 = vBIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<float> imp_block = std::get<0>(result2);
    std::vector<int> imp_block_index = std::get<1>(result2);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BLOCK " << duration.count() << " ms" << std::endl;

    auto imp_norm = vector_norm(imp);

    auto imp_aamp_norm = compute_vector_difference_norm(imp, imp_aamp);
    auto imp_block_norm = compute_vector_difference_norm(imp, imp_block);

    std::cout << "Absolute error AAMP: " << imp_aamp_norm << std::endl;
    std::cout << "Absolute error BLOCK: " << imp_block_norm << std::endl;

    std::cout << "Relative error AAMP: " << imp_aamp_norm / imp_norm << std::endl;
    std::cout << "Relative error BLOCK: " << imp_block_norm / imp_norm << std::endl;
}

auto compute_era(bool daily)
{
    std::cout << "Era5" << std::endl;
    std::vector<float> data;
    std::cout << "Reading data" << std::endl;
    std::string path = "../Data/ERA5_land/";
    std::string filename = path + "/daily_series.txt";
    std::string period_starts_filename = path + "/periods_start_daily.txt";

    if (!daily)
    {
        filename = path + "/hourly_series.txt";
        period_starts_filename = path + "/periods_start_hourly.txt";
    }

    readFile<float>(filename.c_str(), data, "%f");
    // Define the window size, exclude value, and seasons
    const int window_size = 7;
    const int exclude = 4;
    const int interval_length = 90;
    const int n = data.size() - window_size + 1;
    std::cout << "Data size: " << data.size() << ", window size: " << window_size << std::endl;

    std::vector<int> period_starts;
    readFile<int>(period_starts_filename.c_str(), period_starts, "%d");

    std::vector<double> double_data = std::vector<double>(data.begin(), data.end());
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = interval_matrix_profile_brute_force(double_data, window_size, period_starts, interval_length, exclude);
    std::vector<double> double_imp = std::get<0>(result);
    std::vector<float> imp = std::vector<float>(double_imp.begin(), double_imp.end());
    std::vector<int> imp_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BF " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    auto result1 = modified_AAMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<float> imp_aamp = std::get<0>(result1);
    std::vector<int> imp_aamp_index = std::get<1>(result1);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "aamp " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    auto result2 = vBIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<float> imp_block = std::get<0>(result2);
    std::vector<int> imp_block_index = std::get<1>(result2);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BLOCK " << duration.count() << " ms" << std::endl;

    auto imp_norm = vector_norm(imp);

    auto imp_aamp_norm = compute_vector_difference_norm(imp, imp_aamp);
    auto imp_block_norm = compute_vector_difference_norm(imp, imp_block);

    float max_error_aamp;
    float max_error_block;
    float max_error_aamp_rel;
    float max_error_block_rel;

    for (int i = 0; i < double_imp.size(); ++i)
    {
        max_error_aamp = std::max(max_error_aamp, std::abs(imp[i] - imp_aamp[i]));
        max_error_block = std::max(max_error_block, std::abs(imp[i] - imp_block[i]));
        max_error_aamp_rel = std::max(max_error_aamp_rel, std::abs(imp[i] - imp_aamp[i]) / imp[i]);
        max_error_block_rel = std::max(max_error_block_rel, std::abs(imp[i] - imp_block[i]) / imp[i]);
    }

    std::cout << "Component-wise max error AAMP: " << max_error_aamp << std::endl;
    std::cout << "Component-wise max error BLOCK: " << max_error_block << std::endl;
    std::cout << "Component-wise max error AAMP relative: " << max_error_aamp_rel << std::endl;
    std::cout << "Component-wise max error BLOCK relative: " << max_error_block_rel << std::endl;


    std::cout << "Absolute error AAMP: " << imp_aamp_norm << std::endl;
    std::cout << "Absolute error BLOCK: " << imp_block_norm << std::endl;

    std::cout << "Relative error AAMP: " << imp_aamp_norm / imp_norm << std::endl;
    std::cout << "Relative error BLOCK: " << imp_block_norm / imp_norm << std::endl;
}

auto compute_era_dp(bool daily)
{
    std::cout << "Era5 Double Precision" << std::endl;
    std::vector<double> data;
    std::cout << "Reading data" << std::endl;
    std::string path = "../Data/ERA5_land/";
    std::string filename = path + "/daily_series.txt";
    std::string period_starts_filename = path + "/periods_start_daily.txt";

    if (!daily)
    {
        filename = path + "/hourly_series.txt";
        period_starts_filename = path + "/periods_start_hourly.txt";
    }

    readFile<double>(filename.c_str(), data, "%lf");
    // Define the window size, exclude value, and seasons
    const int window_size = 7;
    const int exclude = 4;
    const int interval_length = 90;
    const int n = data.size() - window_size + 1;
    std::cout << "Data size: " << data.size() << ", window size: " << window_size << std::endl;

    std::vector<int> period_starts;
    readFile<int>(period_starts_filename.c_str(), period_starts, "%d");

    std::vector<double> double_data = std::vector<double>(data.begin(), data.end());
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = interval_matrix_profile_brute_force(double_data, window_size, period_starts, interval_length, exclude);
    std::vector<double> double_imp = std::get<0>(result);
    std::vector<float> imp = std::vector<float>(double_imp.begin(), double_imp.end());
    std::vector<int> imp_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BF " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    auto result1 = modified_AAMP(double_data, window_size, period_starts, interval_length, exclude);
    std::vector<double> double_imp_aamp = std::get<0>(result1);
    std::vector<float> imp_aamp = std::vector<float>(double_imp_aamp.begin(), double_imp_aamp.end());
    std::vector<int> imp_aamp_index = std::get<1>(result1);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "aamp " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    auto result2 = vBIMP(double_data, window_size, period_starts, interval_length, exclude);
    std::vector<double> double_imp_block = std::get<0>(result2);
    std::vector<float> imp_block = std::vector<float>(double_imp_block.begin(), double_imp_block.end());
    std::vector<int> imp_block_index = std::get<1>(result2);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BLOCK " << duration.count() << " ms" << std::endl;

    auto imp_norm = vector_norm(imp);

    auto imp_aamp_norm = compute_vector_difference_norm(double_imp, double_imp_aamp);
    auto imp_block_norm = compute_vector_difference_norm(double_imp, double_imp_block);

    // Maximum component-wise absolute error
    double max_error_aamp;
    double max_error_block;

    for (int i = 0; i < double_imp.size(); ++i)
    {
        max_error_aamp = std::max(max_error_aamp, std::abs(double_imp[i] - double_imp_aamp[i]));
        max_error_block = std::max(max_error_block, std::abs(double_imp[i] - double_imp_block[i]));
    }

    std::cout << "Component-wise max error AAMP: " << max_error_aamp << std::endl;
    std::cout << "Component-wise max error BLOCK: " << max_error_block << std::endl;

    std::cout << "Absolute error AAMP: " << imp_aamp_norm << std::endl;
    std::cout << "Absolute error BLOCK: " << imp_block_norm << std::endl;

    std::cout << "Relative error AAMP: " << imp_aamp_norm / imp_norm << std::endl;
    std::cout << "Relative error BLOCK: " << imp_block_norm / imp_norm << std::endl;
}

auto compute_era_daily_s()
{
    std::cout << "Era5" << std::endl;
    std::vector<float> data;
    std::cout << "Reading data" << std::endl;
    std::string path = "../Data/ERA5_land/";
    std::string filename = path + "/hourly_series.txt";
    std::string period_starts_filename = path + "/periods_start_hourly.txt";

    readFile<float>(filename.c_str(), data, "%f");
    // Define the window size, exclude value, and seasons
    const int window_size = 7;
    const int exclude = 4;
    const int interval_length = 90;
    const int n = data.size() - window_size + 1;
    std::cout << "Data size: " << data.size() << ", window size: " << window_size << std::endl;

    std::vector<int> period_starts;
    readFile<int>(period_starts_filename.c_str(), period_starts, "%d");

    auto start_time = std::chrono::high_resolution_clock::now();
    auto result2 = vBIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<float> imp_block = std::get<0>(result2);
    std::vector<int> imp_block_index = std::get<1>(result2);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BLOCK " << duration.count() << " ms" << std::endl;
}

auto compute_era_daily_dp()
{
    std::cout << "Era5" << std::endl;
    std::vector<double> data;
    std::cout << "Reading data" << std::endl;
    std::string path = "../Data/ERA5_land/";
    std::string filename = path + "/hourly_series.txt";
    std::string period_starts_filename = path + "/periods_start_hourly.txt";

    readFile<double>(filename.c_str(), data, "%lf");
    // Define the window size, exclude value, and seasons
    const int window_size = 7;
    const int exclude = 4;
    const int interval_length = 90;
    const int n = data.size() - window_size + 1;
    std::cout << "Data size: " << data.size() << ", window size: " << window_size << std::endl;

    std::vector<int> period_starts;
    readFile<int>(period_starts_filename.c_str(), period_starts, "%d");

    auto start_time = std::chrono::high_resolution_clock::now();
    auto result2 = vBIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_block = std::get<0>(result2);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BLOCK " << duration.count() << " ms" << std::endl;
}

auto compute_synthetic(const int n_year, const int window_size, const int interval_length, const int year_length)
{
    std::vector<float> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    std::vector<int> period_starts;
    for (int year = 0; year < n_year; ++year)
    {
        for (int i = 0; i < year_length; ++i)
        {
            data.push_back(dis(gen));
        }
        period_starts.push_back(year * year_length);
    }
    std::cout << "Data size: " << data.size() << std::endl;

    const int exclude = window_size;

    std::vector<double> double_data = std::vector<double>(data.begin(), data.end());
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = interval_matrix_profile_brute_force(double_data, window_size, period_starts, interval_length, exclude);
    std::vector<double> double_imp = std::get<0>(result);
    std::vector<float> imp = std::vector<float>(double_imp.begin(), double_imp.end());
    std::vector<int> imp_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BF " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    auto result1 = modified_AAMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<float> imp_aamp = std::get<0>(result1);
    std::vector<int> imp_aamp_index = std::get<1>(result1);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "aamp " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    auto result2 = vBIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<float> imp_block = std::get<0>(result2);
    std::vector<int> imp_block_index = std::get<1>(result2);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BLOCK " << duration.count() << " ms" << std::endl;

    auto imp_norm = vector_norm(imp);

    auto imp_aamp_norm = compute_vector_difference_norm(imp, imp_aamp);
    auto imp_block_norm = compute_vector_difference_norm(imp, imp_block);

    std::cout << "Absolute error AAMP: " << imp_aamp_norm << std::endl;
    std::cout << "Absolute error BLOCK: " << imp_block_norm << std::endl;

    std::cout << "Relative error AAMP: " << imp_aamp_norm / imp_norm << std::endl;
    std::cout << "Relative error BLOCK: " << imp_block_norm / imp_norm << std::endl;
}

int main()
{
    // compute_cp("SST");
    // compute_cp("T2M");

    compute_era(true);
    compute_era_dp(true);

    // compute_synthetic(100, 7, 90, 365);
    // compute_synthetic(200, 30, 90, 365);

    // compute_era_daily_s();
    // compute_era_daily_dp();

    return 0;
}
