#include "gtest/gtest.h"
#include "seasonal_matrix_profile.hpp"
#include "utils.hpp"

TEST(SeasonalMatrixProfileTest, SameResult_brute_force_sst)
{
    std::vector<double> data;
    readFile<double>("../Data/tests/ts_sst.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    int window_size = 7;
    int exclude = 4;
    const int n = data.size() - window_size + 1;
    std::vector<std::vector<std::pair<int, int>>> seasons;
    seasons.push_back(build_season_vector<double>("../Data/tests/seasons_sst_0.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/tests/seasons_sst_1.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/tests/seasons_sst_2.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/tests/seasons_sst_3.txt", n));

    // Calculate the matrix profile using the three methods
    auto result_brute_force_blocking = seasonal_matrix_profile_brute_force_blocking(data, window_size, exclude, seasons);
    std::vector<double> seasonal_matrix_profile_brute_force_blocking = std::get<0>(result_brute_force_blocking);
    std::vector<int> index_profile_brute_force_blocking = std::get<1>(result_brute_force_blocking);
    auto result_bf = seasonal_matrix_profile_brute_force(data, window_size, exclude, seasons);
    std::vector<double> seasonal_matrix_profile_bf = std::get<0>(result_bf);
    std::vector<int> index_profile_bf = std::get<1>(result_bf);

    for (int i = 0; i < seasonal_matrix_profile_bf.size(); ++i)
    {
        ASSERT_NEAR(seasonal_matrix_profile_brute_force_blocking[i], seasonal_matrix_profile_bf[i], 1e-10) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", bf_blocking: " << index_profile_brute_force_blocking[i];
    }
}
TEST(SeasonalMatrixProfileTest, SameResult_blocking_sst)
{
    std::vector<double> data;
    readFile<double>("../Data/tests/ts_sst.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    int window_size = 7;
    int exclude = 4;
    const int n = data.size() - window_size + 1;
    std::vector<std::vector<std::pair<int, int>>> seasons;
    seasons.push_back(build_season_vector<double>("../Data/tests/seasons_sst_0.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/tests/seasons_sst_1.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/tests/seasons_sst_2.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/tests/seasons_sst_3.txt", n));

    // Calculate the matrix profile using the three methods
    auto result_brute_force_blocking = seasonal_matrix_profile_brute_force_blocking(data, window_size, exclude, seasons);
    std::vector<double> seasonal_matrix_profile_brute_force_blocking = std::get<0>(result_brute_force_blocking);
    auto result_stomp = seasonal_matrix_profile_STOMP_blocking(data, window_size, exclude, seasons);
    std::vector<double> seasonal_matrix_profile_STOMP_blocking = std::get<0>(result_stomp);

    for (int i = 0; i < seasonal_matrix_profile_STOMP_blocking.size(); ++i)
    {
        auto bf = seasonal_matrix_profile_brute_force_blocking[i]*seasonal_matrix_profile_brute_force_blocking[i];
        auto stomp = seasonal_matrix_profile_STOMP_blocking[i]*seasonal_matrix_profile_STOMP_blocking[i];
        ASSERT_NEAR(bf, stomp, 1e-15) << "Incorrect mp value at index " << i;
    }
}
