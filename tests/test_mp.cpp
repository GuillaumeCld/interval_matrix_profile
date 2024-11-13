#include "gtest/gtest.h"
#include "block.hpp"
#include "matrix_profile.hpp"
#include <utils.hpp>
#include <vector>
#include <random>
#include <distance.hpp>
#include "test_utils.hpp"
class MpTest : public ::testing::Test
{
protected:
    // You can do set-up work for each test here.
    MpTest() {}

    // You can do clean-up work that doesn't throw exceptions here.
    ~MpTest() override {}

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:
    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }
};

TEST_F(MpTest, BruteForce_sst)
{
    std::vector<double> ts;
    readFile<double>("../Data/tests/ts_sst.txt", ts, "%lf");
    std::vector<double> mp_ref;
    readFile<double>("../Data/tests/mp_sst.txt", mp_ref, "%lf");
    const int window_size = 7;
    const int exclude = 2;
    auto mpOutput = MP_bf(ts, window_size, exclude);
    std::vector<double> matrix_profile_brute_force = std::get<0>(mpOutput);
    std::vector<int> index_profile_brute_force = std::get<1>(mpOutput);

    for (int i = 0; i < matrix_profile_brute_force.size(); ++i)
    {
        EXPECT_NEAR(mp_ref[i], matrix_profile_brute_force[i], 2e-7) << "Incorrect mp value at index " << i;
    }
}


TEST_F(MpTest, Block_MP)
{
    int vector_size = 102;
    std::vector<double> data = generateRandomTimeSeries<double>(vector_size);
    int window_size = 3;
    const int exclude = 2;

    auto BfMpOutput = MP_bf(data, window_size, exclude);
    std::vector<double> matrix_profile_stomp_bf = std::get<0>(BfMpOutput);
    std::vector<int> matrix_profile_index_bf = std::get<1>(BfMpOutput);

    for (int block_width = 10; block_width <= 90; block_width += 10)
    {
        for (int block_height = 10; block_height <= 90; block_height += 10)
        {
            auto BlockMpOutput = BAAMP(data, window_size, block_width, block_height, exclude);
            std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
            std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
            for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
            {
                EXPECT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i << " height" << block_height << " and  width" << block_width << "NN index bd:" << matrix_profile_index_bf[i] << " stomp: " << matrix_profile_index_block[i];
            }
        }
    }
}
