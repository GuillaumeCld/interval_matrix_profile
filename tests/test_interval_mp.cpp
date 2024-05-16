#include "gtest/gtest.h"
#include "interval_matrix_profile.hpp"
#include "utils.hpp"

TEST(IntervalMatrixProfileTest, SameResult_bf_stomp_sst)
{
    std::vector<double> data;
    readFile<double>("../Data/ts_sst.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    int window_size = 7;
    int exclude = 4;
    int interval_length = 30;
    const int n = data.size() - window_size + 1;
    std::vector<int> period_starts;
    readFile<int>("../Data/periods_start_sst.txt", period_starts, "%d"); 

    // Calculate the matrix profile using the three methods
    auto result_brute_force = inteval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
    auto result_stomp = inteval_matrix_profile_STOMP_bf(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result_stomp);
    std::vector<int> index_profile_stomp = std::get<1>(result_stomp);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        ASSERT_NEAR(imp_bf[i]*imp_bf[i], imp_stomp[i]*imp_stomp[i], 1e-14) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_stomp[i];
    }
}


// TEST(IntervalMatrixProfileTest, SameResult_stomp_sst)
// {
//     std::vector<double> data;
//     readFile<double>("../Data/ts_sst.txt", data, "%lf");
//     // Define the window size, exclude value, and seasons
//     int window_size = 7;
//     int exclude = 4;
//     int interval_length = 30;
//     const int n = data.size() - window_size + 1;
//     std::vector<int> period_starts;
//     readFile<int>("../Data/periods_start_sst.txt", period_starts, "%d"); 

//     // Calculate the matrix profile using the three methods
//     auto result_brute_force = inteval_matrix_profile_STOMP_bf(data, window_size, period_starts, interval_length, exclude);
//     std::vector<double> imp_bf = std::get<0>(result_brute_force);
//     std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
//     auto result_stomp = inteval_matrix_profile_STOMP(data, window_size, period_starts, interval_length, exclude);
//     std::vector<double> imp_stomp = std::get<0>(result_stomp);
//     std::vector<int> index_profile_stomp = std::get<1>(result_stomp);

//     for (int i = 0; i < imp_stomp.size(); ++i)
//     {
//         ASSERT_NEAR(imp_bf[i], imp_stomp[i], 1e-10) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_stomp[i];
//         // printf("value %f %f\n", imp_bf[i], imp_stomp[i]);
//     }
// }