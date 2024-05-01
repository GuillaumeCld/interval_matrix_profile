#include "gtest/gtest.h"
#include "block.hpp"
#include "matrix_profile.hpp"
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

TEST_F(MpTest, Block)
{
    int vector_size = 10002;
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    std::cout << "Random gen" << std::endl;

    for (int i = 0; i < vector_size; ++i)
    {
        data.push_back(dis(gen));
    }

    int window_size = 3;

    printf("Brute force\n");
    auto start_bf = std::chrono::high_resolution_clock::now();
    auto BfMpOutput = computeMatrixProfileBruteForce(data, window_size);
    auto end_bf = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_bf = end_bf - start_bf;
    std::cout << "Brute force computation time: " << elapsed_bf.count() << " seconds" << std::endl;
    std::vector<double> matrix_profile_stomp_bf = std::get<0>(BfMpOutput);
    std::vector<int> matrix_profile_index_bf = std::get<1>(BfMpOutput);

    // for (int block_width = 10; block_width <= 90; block_width += 10)
    // {
    //     for (int block_height = 10; block_height <= 90; block_height += 10)
    //     {
    //         std::cout << "Block STOMP with width: " << block_width << ", height: " << block_height << std::endl;
    //         auto start_block = std::chrono::high_resolution_clock::now();
    //         auto BlockMpOutput = blockSTOMP(data, window_size, block_width, block_height);
    //         auto end_block = std::chrono::high_resolution_clock::now();
    //         std::chrono::duration<double> elapsed_block = end_block - start_block;
    //         std::cout << "Block STOMP computation time: " << elapsed_block.count() << " seconds" << std::endl;
    //         std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
    //         std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);
    //         for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
    //         {
    //             EXPECT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i << " height" << block_height << " and  width" << block_width;
    //         }
    //     }
    // }

    std::cout << "Block STOMP" << std::endl;
    auto start_block = std::chrono::high_resolution_clock::now();
    auto BlockMpOutput = blockSTOMP(data, window_size, 100, 100);
    auto end_block = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_block = end_block - start_block;
    std::cout << "Block STOMP computation time: " << elapsed_block.count() << " seconds" << std::endl;
    std::vector<double> matrix_profile_stomp_block = std::get<0>(BlockMpOutput);
    std::vector<int> matrix_profile_index_block = std::get<1>(BlockMpOutput);

    for (int i = 0; i < matrix_profile_stomp_bf.size(); ++i)
    {
        // printf("%d BF: %f,%d, Block: %f,%d\n",i, matrix_profile_stomp_bf[i], matrix_profile_index_bf[i], matrix_profile_stomp_block[i], matrix_profile_index_block[i]);
        ASSERT_NEAR(matrix_profile_stomp_bf[i], matrix_profile_stomp_block[i], 1e-10) << "Incorrect mp value at index " << i << " with NN index " << matrix_profile_index_bf[i] << " and " << matrix_profile_index_block[i];
    }
}