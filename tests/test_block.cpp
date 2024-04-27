#include "gtest/gtest.h"
#include "block.hpp"
#include <vector>
#include <random>
#include <distance.hpp>

class BlockTest : public ::testing::Test {
protected:
    // You can do set-up work for each test here.
    BlockTest() {}

    // You can do clean-up work that doesn't throw exceptions here.
    ~BlockTest() override {}

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:
    void SetUp() override {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    // Class members declared here can be used by all tests in the test suite for Block.
};

TEST_F(BlockTest, TestSTOMPParallelogram) {
    int n = 100;
    int m = 3;
    int exclude = 2;

    // Generate a random vector of doubles of size n
    std::vector<double> time_series;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        time_series.push_back(dis(gen));
    }

    int block_width = 10;
    int block_height = 10;
    int block_i = 0;
    int block_j = 0;
    int block_ID = 0;

    std::vector<double> first_row(n-m+1);
    std::span view = std::span(&time_series[0], m);
    for (int j = 0; j < block_width; ++j) {
        const auto distance = dotProduct(view, std::span(&time_series[j], m));
        first_row[j] = distance;
    }

    std::vector<double> intitial_row(block_width);
    for (int j = 0; j < block_width; ++j) {
        intitial_row[j] = first_row[j];
    }
    // Initialize a Block object
    block<double> block(n, m, exclude, block_i, block_j, block_ID, block_width, block_height, first_row, intitial_row, time_series);

    // Assert if the block type is correct
    ASSERT_EQ(block.get_type(), PARALLELOGRAM) << "Block type is not 'parallelogram'";

    // Call the STOMP method
    block.STOMP();

    std::vector<double> block_row = block.get_row();
    std::vector<min_pair<double>> block_local_min_row = block.get_local_min_rows();

    // Assert if the last row of the block is correct
    view = std::span(&time_series[block_height], m);
    for (int j = 0; j < block_width; ++j) {
        int global_j = block_j + block_height + j;
        const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
        EXPECT_EQ(block_row[j], distance) << "Elements of the row's final row are not equal to the explicit calculation";
    }
    // Assert if the local min row is correct
    for (int i = 0; i < block_height; ++i) {
        view = std::span(&time_series[i], m);
        auto min = std::numeric_limits<double>::max();
        int argmin = -1;
        for (int j = 0; j < block_width; ++j) {
            int global_j = block_j + block_height + j;
            const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
            if (distance < min) {
                min = distance;
                argmin = j;
            }
        }
        EXPECT_EQ(block_local_min_row[i].value, min) << "Values of the local min row are not equal to the explicit calculation";
        EXPECT_EQ(block_local_min_row[i].index, argmin) << "Indices of the local min row are not equal to the explicit calculation";     
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}