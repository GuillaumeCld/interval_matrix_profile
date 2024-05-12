#include "gtest/gtest.h"
#include "seasonal_matrix_profile.hpp"

TEST(SeasonalMatrixProfileTest, SameResult)
{
    // Generate some test data
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    // Define the window size, exclude value, and seasons
    int window_size = 3;
    int exclude = 1;
    std::vector<std::vector<std::pair<int, int>>> seasons = {
        {{0, 2}, {3, 5}, {6, 8}}};

    // Calculate the matrix profile using the three methods
    auto result_brute_force_blocking = seasonal_matrix_profile_brute_force_blocking(data, window_size, exclude, seasons);
    auto result_brute_force = seasonal_matrix_profile_brute_force(data, window_size, exclude, seasons);
    auto result_stomp = seasonal_matrix_profile_STOMP_blocking(data, window_size, exclude, seasons);

    // Check if the results are the same
    ASSERT_EQ(result_brute_force_blocking, result_brute_force);
    ASSERT_EQ(result_brute_force_blocking, result_stomp);
}