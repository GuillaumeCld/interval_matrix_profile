#include "gtest/gtest.h"
#include "block.hpp"
#include "matrix_profile.hpp"
#include <utils.hpp>
#include <vector>
#include <random>
#include <distance.hpp>

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

    // Class members declared here can be used by all tests in the test suite for Block.
};

TEST_F(MpTest, BruteForce_sst)
{
    std::vector<double> ts;
    readFile<double>("../Data/ts_sst.txt", ts, "%lf");
    std::vector<double> mp_ref;
    readFile<double>("../Data/mp_sst.txt", mp_ref, "%lf");
    int window_size = 7;

    auto mpOutput = computeMatrixProfileBruteForce(ts, window_size);
    std::vector<double> matrix_profile_brute_force = std::get<0>(mpOutput);
    std::vector<int> index_profile_brute_force = std::get<1>(mpOutput);

    for (int i = 0; i < matrix_profile_brute_force.size(); ++i)
    {
        EXPECT_NEAR(mp_ref[i], matrix_profile_brute_force[i], 2e-7) << "Incorrect mp value at index " << i;
    }
}

TEST_F(MpTest, STOMP_sst)
{
    std::vector<double> ts;
    readFile<double>("../Data/ts_sst.txt", ts, "%lf");
    std::vector<double> mp_ref;
    readFile<double>("../Data/mp_sst.txt", mp_ref, "%lf");
    int window_size = 7;

    auto mpOutput = blockSTOMP_v2(ts, window_size, 100, 100);
    std::vector<double> matrix_profile = std::get<0>(mpOutput);
    std::vector<int> index_profile = std::get<1>(mpOutput);

    for (int i = 0; i < matrix_profile.size(); ++i)
    {
        EXPECT_NEAR(mp_ref[i], matrix_profile[i], 2e-7) << "Incorrect mp value at index " << i;
    }
}

TEST_F(MpTest, Block)
{
    int vector_size = 1002;
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    for (int i = 0; i < vector_size; ++i)
    {
        data.push_back(dis(gen));
    }

    int window_size = 3;

    auto BfMpOutput = computeMatrixProfileBruteForce(data, window_size);
    std::vector<double> matrix_profile_stomp_bf = std::get<0>(BfMpOutput);
    std::vector<int> matrix_profile_index_bf = std::get<1>(BfMpOutput);

    for (int block_width = 10; block_width <= 90; block_width += 10)
    {
        for (int block_height = 10; block_height <= 90; block_height += 10)
        {
            // std::cout << "Block STOMP with width: " << block_width << ", height: " << block_height << std::endl;
            auto BlockMpOutput = blockSTOMP(data, window_size, block_width, block_height);
            std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
            std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
            for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
            {
                EXPECT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i << " height" << block_height << " and  width" << block_width << "NN index bd:" << matrix_profile_index_bf[i] << " stomp: " << matrix_profile_index_block[i];
            }
        }
    }
}

TEST_F(MpTest, Block_v2)
{
    int vector_size = 102;
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    for (int i = 0; i < vector_size; ++i)
    {
        data.push_back(dis(gen));
    }

    int window_size = 3;

    auto BfMpOutput = computeMatrixProfileBruteForce(data, window_size);
    std::vector<double> matrix_profile_stomp_bf = std::get<0>(BfMpOutput);
    std::vector<int> matrix_profile_index_bf = std::get<1>(BfMpOutput);

    for (int block_width = 10; block_width <= 90; block_width += 10)
    {
        for (int block_height = 10; block_height <= 90; block_height += 10)
        {
            auto BlockMpOutput = blockSTOMP_v2(data, window_size, block_width, block_height);
            std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
            std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
            for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
            {
                EXPECT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i << " height" << block_height << " and  width" << block_width << "NN index bd:" << matrix_profile_index_bf[i] << " stomp: " << matrix_profile_index_block[i];
            }
        }
    }
    // auto BlockMpOutput = blockSTOMP_v2(data, window_size, 10, 60);
    // std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
    // std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
    // for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
    // {
    //     EXPECT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i;
    // }
}

TEST_F(MpTest, Block_v3)
{
    int vector_size = 1002;
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    for (int i = 0; i < vector_size; ++i)
    {
        data.push_back(dis(gen));
    }

    int window_size = 3;

    auto BfMpOutput = computeMatrixProfileBruteForce(data, window_size);
    std::vector<double> matrix_profile_stomp_bf = std::get<0>(BfMpOutput);
    std::vector<int> matrix_profile_index_bf = std::get<1>(BfMpOutput);

    for (int block_width = 10; block_width <= 90; block_width += 10)
    {
        for (int block_height = 10; block_height <= 90; block_height += 10)
        {
            auto BlockMpOutput = blockSTOMP_v3(data, window_size, block_width, block_height);
            std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
            std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
            for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
            {
                EXPECT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i << " height" << block_height << " and  width" << block_width;
            }
        }
    }
    // auto BlockMpOutput = blockSTOMP_v3(data, window_size, 10, 2);
    // std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
    // std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
    // for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
    // {
    //     EXPECT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i;
    // }
}

TEST_F(MpTest, Block_v4)
{
    int vector_size = 1002;
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    for (int i = 0; i < vector_size; ++i)
    {
        data.push_back(dis(gen));
    }

    int window_size = 3;

    auto BfMpOutput = computeMatrixProfileBruteForce(data, window_size);
    std::vector<double> matrix_profile_stomp_bf = std::get<0>(BfMpOutput);
    std::vector<int> matrix_profile_index_bf = std::get<1>(BfMpOutput);

    for (int block_width = 10; block_width <= 90; block_width += 10)
    {
        for (int block_height = 10; block_height <= 90; block_height += 10)
        {
            auto BlockMpOutput = blockSTOMP_v4(data, window_size, block_width, block_height);
            std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
            std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
            for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
            {
                EXPECT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i << " height" << block_height << " and  width" << block_width;
            }
        }
    }
    auto BlockMpOutput = blockSTOMP_v4(data, window_size, 50, 30);
    std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
    std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
    for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
    {
        ASSERT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i;
    }
}

TEST_F(MpTest, Block_v5)
{
    int vector_size = 1002;
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    for (int i = 0; i < vector_size; ++i)
    {
        data.push_back(dis(gen));
    }

    int window_size = 3;

    auto BfMpOutput = computeMatrixProfileBruteForce(data, window_size);
    std::vector<double> matrix_profile_stomp_bf = std::get<0>(BfMpOutput);
    std::vector<int> matrix_profile_index_bf = std::get<1>(BfMpOutput);

    for (int block_width = 10; block_width <= 90; block_width += 10)
    {
        for (int block_height = 10; block_height <= 90; block_height += 10)
        {
            auto BlockMpOutput = blockSTOMP_v5(data, window_size, block_width, block_height);
            std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
            std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
            for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
            {
                EXPECT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i << " height" << block_height << " and  width" << block_width;
            }
        }
    }
    auto BlockMpOutput = blockSTOMP_v5(data, window_size, 50, 30);
    std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
    std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
    for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
    {
        ASSERT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i;
    }
}