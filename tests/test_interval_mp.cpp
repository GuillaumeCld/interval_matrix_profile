#include "gtest/gtest.h"
#include "interval_matrix_profile.hpp"
#include "bimp.hpp"
#include "bimb_knn.hpp"
#include "utils.hpp"
#include <random> 

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
    auto result_brute_force = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
    auto result_stomp = BIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result_stomp);
    std::vector<int> index_profile_stomp = std::get<1>(result_stomp);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        ASSERT_NEAR(imp_bf[i] * imp_bf[i], imp_stomp[i] * imp_stomp[i], 1e-14) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_stomp[i] << " with the values: " << imp_bf[i] << " and " << imp_stomp[i];
    }
}
TEST(IntervalMatrixProfileTest, SameResult_bf_stomp_ini_sst)
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
    auto result_brute_force = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
    auto result_stomp = BIMP_intialized(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result_stomp);
    std::vector<int> index_profile_stomp = std::get<1>(result_stomp);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        ASSERT_NEAR(imp_bf[i] * imp_bf[i], imp_stomp[i] * imp_stomp[i], 1e-14) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_stomp[i] << "with the values: " << imp_bf[i] << " and " << imp_stomp[i];
    }
}

TEST(IntervalMatrixProfileTest, SameResult_stomp_sst)
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
    auto result_stomp_init = BIMP_intialized(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_ini = std::get<0>(result_stomp_init);
    std::vector<int> index_profile_ini = std::get<1>(result_stomp_init);
    auto result_stomp = BIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result_stomp);
    std::vector<int> index_profile_stomp = std::get<1>(result_stomp);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        ASSERT_NEAR(imp_ini[i] * imp_ini[i], imp_stomp[i] * imp_stomp[i], 1e-14) << "Incorrect mp value at index " << i << ", bf: " << index_profile_ini[i] << ", STOMP: " << index_profile_stomp[i];
    }
}

TEST(IntervalMatrixProfileTest, SameResult_bf_modif_stomp_sst)
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
    auto result_brute_force = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
    auto result_stomp = modified_AAMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result_stomp);
    std::vector<int> index_profile_stomp = std::get<1>(result_stomp);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        ASSERT_NEAR(imp_bf[i] * imp_bf[i], imp_stomp[i] * imp_stomp[i], 1e-13) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_stomp[i] << " with the values: " << imp_bf[i] << " and " << imp_stomp[i];
    }
}

TEST(IntervalMatrixProfileTest, SameResult_1NN)
{
    std::vector<double> data;
    readFile<double>("../Data/ts_sst.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    const int k = 1;
    int window_size = 7;
    int exclude = 4;
    int interval_length = 30;
    const int n = data.size() - window_size + 1;
    std::vector<int> period_starts;
    readFile<int>("../Data/periods_start_sst.txt", period_starts, "%d");

    // Calculate the matrix profile using the three methods
    auto result_brute_force = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
    auto result = BIMP_kNN(data, window_size, period_starts, interval_length, exclude, k, false);
    std::vector<double> imp_stomp = std::get<0>(result);
    std::vector<int> index_profile_stomp = std::get<1>(result);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        ASSERT_NEAR(imp_bf[i] * imp_bf[i], imp_stomp[i] * imp_stomp[i], 1e-13) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_stomp[i] << " with the values: " << imp_bf[i] << " and " << imp_stomp[i];
    }
}


TEST(IntervalMatrixProfileTest, SameResult_bf_1NN)
{
    std::vector<double> data;
    readFile<double>("../Data/ts_sst.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    const int k = 1;
    int window_size = 7;
    int exclude = 4;
    int interval_length = 30;
    const int n = data.size() - window_size + 1;
    std::vector<int> period_starts;
    readFile<int>("../Data/periods_start_sst.txt", period_starts, "%d");

    // Calculate the matrix profile using the three methods
    auto result_brute_force = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
    auto result = interval_matrix_profile_brute_force_knn(data, window_size, period_starts, interval_length, exclude, k);
    std::vector<double> imp_1 = std::get<0>(result);
    std::vector<int> index_profile_1 = std::get<1>(result);

    for (int i = 0; i < imp_1.size(); ++i)
    {
        ASSERT_NEAR(imp_bf[i] * imp_bf[i], imp_1[i] * imp_1[i], 1e-13) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_1[i] << " with the values: " << imp_bf[i] << " and " << imp_1[i];
    }
}

TEST(IntervalMatrixProfileTest, GreaterDistance_2NN)
{
    std::vector<double> data;
    readFile<double>("../Data/ts_sst.txt", data, "%lf");
    // Define the window size, exclude value, and seasons
    const int k = 2;
    int window_size = 7;
    int exclude = 4;
    int interval_length = 30;
    const int n = data.size() - window_size + 1;
    std::vector<int> period_starts;
    readFile<int>("../Data/periods_start_sst.txt", period_starts, "%d");

    // Calculate the matrix profile using the three methods
    auto result_brute_force = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
    auto result = BIMP_kNN(data, window_size, period_starts, interval_length, exclude, k, false);
    std::vector<double> imp_stomp = std::get<0>(result);
    std::vector<int> index_profile_stomp = std::get<1>(result);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        auto diff = imp_stomp[i] * imp_stomp[i] - imp_bf[i] * imp_bf[i];
        if (std::abs(diff) <= 1e-13)
        {
            continue;
        }
        else
        {
            ASSERT_GT(diff, 0.0) << "2NN is lesser than 1NN " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_stomp[i] << " with the values: " << imp_bf[i] << " and " << imp_stomp[i];
        }
    }
}

TEST(IntervalMatrixProfileTest, SameResult_kNN)
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

    for (int k = 1; k < 20; ++k)
    {
        // Calculate the matrix profile using the three methods
        auto result_brute_force = interval_matrix_profile_brute_force_knn(data, window_size, period_starts, interval_length, exclude,k);
        std::vector<double> imp_bf = std::get<0>(result_brute_force);
        std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
        auto result = BIMP_kNN(data, window_size, period_starts, interval_length, exclude, k, false);
        std::vector<double> imp_stomp = std::get<0>(result);
        std::vector<int> index_profile_stomp = std::get<1>(result);

        for (int i = 0; i < imp_stomp.size(); ++i)
        {
            ASSERT_NEAR(imp_bf[i] * imp_bf[i], imp_stomp[i] * imp_stomp[i], 1e-13) << "For k=" << k << " Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_stomp[i] << " with the values: " << imp_bf[i] << " and " << imp_stomp[i];
        }
    }
}



TEST(IntervalMatrixProfileTest, SameResult_bf_vectorized_sst)
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
    auto result_brute_force = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
    auto result_stomp = vBIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result_stomp);
    std::vector<int> index_profile_stomp = std::get<1>(result_stomp);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        ASSERT_NEAR(imp_bf[i] * imp_bf[i], imp_stomp[i] * imp_stomp[i], 1e-14) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_stomp[i] << " with the values: " << imp_bf[i] << " and " << imp_stomp[i];
    }
}



TEST(IntervalMatrixProfileTest, SameResult_bf_vectorized_synth)
{
    const int n_year = 100;
    const int year_length = 365;
    int window_size = 7;
    int exclude = 4;
    int interval_length = 30;

    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 30);
    std::vector<int> period_starts;
    for (int year = 0; year < n_year; ++year)
    {
        for (int i = 0; i < year_length; ++i)
        {
            data.push_back(dis(gen));
        }
        period_starts.push_back(year * year_length);
    }
    // Define the window size, exclude value, and seasons

    const int n = data.size() - window_size + 1;

    auto result_brute_force = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_bf = std::get<0>(result_brute_force);
    std::vector<int> index_profile_bf = std::get<1>(result_brute_force);
    auto result_stomp = vBIMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result_stomp);
    std::vector<int> index_profile_stomp = std::get<1>(result_stomp);

    for (int i = 0; i < imp_stomp.size(); ++i)
    {
        ASSERT_NEAR(imp_bf[i] * imp_bf[i], imp_stomp[i] * imp_stomp[i], 1e-10) << "Incorrect mp value at index " << i << ", bf: " << index_profile_bf[i] << ", STOMP: " << index_profile_stomp[i] << " with the values: " << imp_bf[i] << " and " << imp_stomp[i];
    }
}