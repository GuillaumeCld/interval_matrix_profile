#include "gtest/gtest.h"
#include "interval_matrix_profile.hpp"
#include "bimp.hpp"
#include "outer_bimp.hpp"
#include "utils.hpp"
#include <random>

TEST(OuterTest, self_join)
{

    std::vector<double> data;
    readFile<double>("../Data/tests/ts_sst.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    int window_size = 7;
    int exclude = -1;
    int interval_length = 30;
    const int n = data.size() - window_size + 1;
    std::vector<int> period_starts;
    readFile<int>("../Data/tests/periods_start_sst.txt", period_starts, "%d");

    // Calculate the matrix profile using the three methods
    auto result_brute_force = imp_bf(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);

    auto result_outer = outer_BIMP(data, data, window_size, period_starts, period_starts, interval_length);
    std::vector<double> imp_stomp = std::get<0>(result_outer);
    std::vector<int> index_profile_stomp = std::get<1>(result_outer);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        ASSERT_NEAR(imp_bf[i] * imp_bf[i], imp_stomp[i] * imp_stomp[i], 1e-14) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", Outer: " << index_profile_stomp[i] << " with the values: " << imp_bf[i] << " and " << imp_stomp[i];
    }
}

TEST(OuterTest, AB_join_bf)
{
    std::vector<double> data_A;
    std::vector<double> data_B;
    readFile<double>("../Data/tests/ts_sst.txt", data_A, "%lf");
    readFile<double>("../Data/tests/ts_sst.txt", data_B, "%lf");
    // Define the window size, exclude value, and seasons
    int window_size = 7;
    int interval_length = 30;
    const int n_A = data_A.size() - window_size + 1;
    const int n_B = data_B.size() - window_size + 1;
    std::vector<int> period_starts_A;
    std::vector<int> period_starts_B;
    readFile<int>("../Data/tests/periods_start_sst.txt", period_starts_A, "%d");
    readFile<int>("../Data/tests/periods_start_sst.txt", period_starts_B, "%d");

    // Randomly change of the values in data_B
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto &val : data_B)
    {
        val += dist(rng);
    }

    auto result_brute_force = imp_bf_ab(data_A, data_B, window_size, period_starts_A, period_starts_B, interval_length);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);

    auto result_outer = outer_BIMP(data_A, data_B, window_size, period_starts_A, period_starts_B, interval_length);
    std::vector<double> imp_stomp = std::get<0>(result_outer);
    std::vector<int> index_profile_stomp = std::get<1>(result_outer);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        ASSERT_NEAR(imp_bf[i] * imp_bf[i], imp_stomp[i] * imp_stomp[i], 1e-12) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", Outer: " << index_profile_stomp[i] << " with the values: " << imp_bf[i] << " and " << imp_stomp[i];
    }
}