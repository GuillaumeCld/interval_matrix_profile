#include "gtest/gtest.h"
#include "seasonal_matrix_profile.hpp"
#include "utils.hpp"

TEST(SeasonalMatrixProfileTest, SameResult_brute_force_sst)
{
    std::vector<double> data;
    readFile<double>("../Data/ts_sst.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    int window_size = 7;
    int exclude = 4;
    const int n = data.size() - window_size + 1;
    std::vector<std::vector<std::pair<int, int>>> seasons;
    seasons.push_back(build_season_vector<double>("../Data/seasons_sst_0.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/seasons_sst_1.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/seasons_sst_2.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/seasons_sst_3.txt", n));

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
    readFile<double>("../Data/ts_sst.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    int window_size = 7;
    int exclude = 4;
    const int n = data.size() - window_size + 1;
    std::vector<std::vector<std::pair<int, int>>> seasons;
    seasons.push_back(build_season_vector<double>("../Data/seasons_sst_0.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/seasons_sst_1.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/seasons_sst_2.txt", n));
    seasons.push_back(build_season_vector<double>("../Data/seasons_sst_3.txt", n));

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

// TEST(SeasonalMatrixProfileTest, BruteForce_sst)
// {
//     std::vector<double> ts;
//     readFile<double>("../Data/ts_sst.txt", ts, "%lf");
//     std::vector<double> mp_ref;
//     readFile<double>("../Data/smp_sst.txt", mp_ref, "%lf");
//     const int window_size = 7;
//     const int exclude = 2;
//     const int n = ts.size() - window_size + 1;
//     std::vector<std::vector<std::pair<int, int>>> seasons;
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_0.txt", n));
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_1.txt", n));
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_2.txt", n));
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_3.txt", n));

//     auto mpOutput = seasonal_matrix_profile_brute_force(ts, window_size, exclude, seasons);
//     std::vector<double> matrix_profile_brute_force = std::get<0>(mpOutput);
//     std::vector<int> index_profile_brute_force = std::get<1>(mpOutput);

//     for (int i = 0; i < matrix_profile_brute_force.size(); ++i)
//     {
//         EXPECT_NEAR(mp_ref[i], matrix_profile_brute_force[i], 2e-7) << "Incorrect mp value at index " << i;
//     }
// }

// TEST(SeasonalMatrixProfileTest, BruteForce_blocking_sst)
// {
//     std::vector<double> ts;
//     readFile<double>("../Data/ts_sst.txt", ts, "%lf");
//     std::vector<double> mp_ref;
//     readFile<double>("../Data/smp_sst.txt", mp_ref, "%lf");
//     std::vector<int> mp_ind_ref;
//     readFile<int>("../Data/smp_ind_sst.txt", mp_ind_ref, "%d");
//     const int window_size = 7;
//     const int exclude = 4;
//     const int n = ts.size() - window_size + 1;
//     std::vector<std::vector<std::pair<int, int>>> seasons;
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_0.txt", n));
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_1.txt", n));
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_2.txt", n));
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_3.txt", n));

//     auto mpOutput = seasonal_matrix_profile_brute_force_blocking(ts, window_size, exclude, seasons);
//     std::vector<double> matrix_profile_brute_force = std::get<0>(mpOutput);
//     std::vector<int> index_profile_brute_force = std::get<1>(mpOutput);

//     for (int i = 0; i < matrix_profile_brute_force.size(); ++i)
//     {
//         EXPECT_NEAR(mp_ref[i], matrix_profile_brute_force[i], 5e-6) << "Incorrect mp value at index " << i << ", ref: " << mp_ind_ref[i] << " calc: " << index_profile_brute_force[i];;
//     }
// }

// TEST(SeasonalMatrixProfileTest, STOMP_blocking_sst)
// {
//     std::vector<double> ts;
//     readFile<double>("../Data/ts_sst.txt", ts, "%lf");
//     std::vector<double> mp_ref;
//     readFile<double>("../Data/smp_sst.txt", mp_ref, "%lf");
//     const int window_size = 7;
//     const int exclude = 2;
//     const int n = ts.size() - window_size + 1;

//     std::vector<std::vector<std::pair<int, int>>> seasons;
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_0.txt", n));
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_1.txt", n));
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_2.txt", n));
//     seasons.push_back(build_season_vector<double>("../Data/seasons_sst_3.txt", n));

//     auto mpOutput = seasonal_matrix_profile_STOMP_blocking(ts, window_size, exclude, seasons);
//     std::vector<double> seasonal_matrix_profile_STOMP_blocking = std::get<0>(mpOutput);
//     std::vector<int> index_profile_stomp = std::get<1>(mpOutput);

//     for (int i = 0; i < seasonal_matrix_profile_STOMP_blocking.size(); ++i)
//     {
//         EXPECT_NEAR(mp_ref[i], seasonal_matrix_profile_STOMP_blocking[i], 2e-6) << "Incorrect mp value at index " << i;
//     }
// }