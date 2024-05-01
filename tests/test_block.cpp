#include "gtest/gtest.h"
#include "block.hpp"
#include <vector>
#include <random>
#include <distance.hpp>

class BlockTest : public ::testing::Test
{
protected:
    // You can do set-up work for each test here.
    BlockTest() {}

    // You can do clean-up work that doesn't throw exceptions here.
    ~BlockTest() override {}

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

TEST_F(BlockTest, Parallelogram)
{
    int n = 100;
    int m = 3;
    int exclude = 2;

    // Generate a random vector of doubles of size n
    std::vector<double> time_series;
    std::mt19937 gen(123); // Use a fixed seed of 123
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < n; i++)
    {
        time_series.push_back(dis(gen));
    }
    int block_width = 10;
    int block_height = 10;
    int block_i = 0;
    int block_j = 0;
    int block_ID = 0;

    std::vector<double> first_row(n - m + 1);
    std::span view = std::span(&time_series[0], m);
    for (int j = 0; j < n - m + 1; ++j)
    {
        const auto distance = dotProduct(view, std::span(&time_series[j], m));
        first_row[j] = distance;
    }

    std::vector<double> initial_row(block_width);
    for (int j = 0; j < block_width; ++j)
    {
        initial_row[j] = first_row[j];
    }
    // Initialize a Block object
    block<double> block(n-m+1, m, exclude, block_i, block_j, block_ID, block_width, block_height, first_row, initial_row, time_series);

    // Assert if the block type is correct
    ASSERT_EQ(block.get_type(), PARALLELOGRAM) << "Block type is not 'parallelogram'";

    // Call the STOMP method
    block.STOMP();

    std::vector<double> block_row = block.get_row();
    std::vector<min_pair<double>> block_local_min_row = block.get_local_min_rows();

    // Assert if the last row of the block is correct
    view = std::span(&time_series[block_height - 1], m);
    for (int j = 0; j < block_width; ++j)
    {
        int global_j = block_j + block_height + j - 1;
        const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
        EXPECT_DOUBLE_EQ(block_row[j], distance) << "Elements of the row's final row are not equal to the explicit calculation";
    }
    // Assert if the local min row is correct
    for (int i = 0; i < block_height; ++i)
    {
        view = std::span(&time_series[i], m);
        auto min = std::numeric_limits<double>::max();
        int argmin = -1;
        for (int j = 0; j < block_width; ++j)
        {
            int global_j = block_j + i + j;
            const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
            if (distance < min and (global_j < block_i + i - exclude or global_j > block_i + i + exclude))
            {
                min = distance;
                argmin = global_j;
            }
        }
        EXPECT_NEAR(block_local_min_row[i].value, min, double(1e-14)) << "Values of the local min row are not equal to the explicit calculation at row " << i;
        EXPECT_EQ(block_local_min_row[i].index, argmin) << "Indices of the local min row are not equal to the explicit calculation";
    }
}

// Test for STOMP_triangle
TEST_F(BlockTest, Triangle)
{

    int n = 100;
    int m = 3;
    int exclude = 2;

    // Generate a random vector of doubles of size n
    std::vector<double> time_series;
    std::mt19937 gen(123); // Use a fixed seed of 123
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < n; i++)
    {
        time_series.push_back(dis(gen));
    }
    int block_width = 10;
    int block_height = 10;
    int block_i = 0;
    int block_j = -11;
    int block_ID = 0;

    std::vector<double> initial_row(block_width);
    std::vector<double> first_row(n - m + 1);
    std::span view = std::span(&time_series[0], m);
    for (int j = 0; j < n - m + 1; ++j)
    {
        const auto distance = dotProduct(view, std::span(&time_series[j], m));
        first_row[j] = distance;
    }

    block<double> block(n-m+1, m, exclude, block_i, block_j, block_ID, block_width, block_height, first_row, initial_row, time_series);

    // Assert if the block type is correct
    ASSERT_EQ(block.get_type(), TRIANGLE) << "Block type is not 'triangle'";
    // Execute
    block.STOMP();
    std::vector<double> block_row = block.get_row();
    std::vector<min_pair<double>> block_local_min_row = block.get_local_min_rows();
    // Verify
    view = std::span(&time_series[block_i + block_height - 1], m);
    int j_start = 0 - (block_j + block_height - 1);
    int j_end = block_j + block_height - 1 + block_width;
    for (int j = 0; j < j_end; ++j)
    {
        int global_j = j;
        const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
        EXPECT_NEAR(block_row[j_start + j], distance, double(1e-14)) << "Elements of the row's final row are not equal to the explicit calculation";
    }

    for (int i = 0; i < block_height; ++i)
    {
        int global_i = block_i + i;
        view = std::span(&time_series[global_i], m);
        auto min = std::numeric_limits<double>::max();
        int argmin = -1;
        for (int j = 0; j < i - 1; ++j)
        {
            // int row_j = block_width - 1 - j;
            const auto distance = dotProduct(view, std::span(&time_series[j], m));
            if (distance < min and (j < block_i + i - exclude or j > block_i + i + exclude))
            {
                min = distance;
                argmin = j;
                // printf("row %d min: %f, argmin: %d\n",i,  min, argmin);
            }
        }
        EXPECT_NEAR(block_local_min_row[i].value, min, double(1e-14)) << "Values of the local min row are not equal to the explicit calculation at row " << i;
        EXPECT_EQ(block_local_min_row[i].index, argmin) << "Indices of the local min row are not equal to the explicit calculation";
    }
}

// Test for STOMP_polygon
TEST_F(BlockTest, Polygon)
{
    int n = 100;
    int m = 3;
    int exclude = 2;

    // Generate a random vector of doubles of size n
    std::vector<double> time_series;
    std::mt19937 gen(123); // Use a fixed seed of 123
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < n; i++)
    {
        time_series.push_back(dis(gen));
    }
    int block_width = 10;
    int block_height = 10;
    int block_i = 0;
    int block_j = -5;
    int block_ID = 0;

    std::vector<double> first_row(n - m + 1);
    std::span view = std::span(&time_series[0], m);
    for (int j = 0; j < n-m+1; ++j)
    {
        const auto distance = dotProduct(view, std::span(&time_series[j], m));
        first_row[j] = distance;
    }

    std::vector<double> initial_row(block_width);
    for (int j = 5; j < block_width; ++j)
    {
        initial_row[j] = first_row[j-5];
    }
    // Initialize a Block object
    block<double> block(n-m+1, m, exclude, block_i, block_j, block_ID, block_width, block_height, first_row, initial_row, time_series);

    // Assert if the block type is correct
    ASSERT_EQ(block.get_type(), POLYGON) << "Block type is not 'polygon'";
    // Call the STOMP method
    // block.STOMP();
    std::vector<double> block_row = block.get_row();
    std::vector<min_pair<double>> block_local_min_row = block.get_local_min_rows();

    // view = std::span(&time_series[block_height - 1], m);
    // for (int j = 0; j < block_width; ++j)
    // {
    //     int global_j = block_j + block_height + j - 1;
    //     const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
    //     EXPECT_NEAR(block_row[j], distance, double(1e-14)) << "Elements of the row's final row are not equal to the explicit calculation";
    // }
    // // Assert if the local min row is correct
    // for (int i = 0; i < block_height; ++i)
    // {
    //     view = std::span(&time_series[i], m);
    //     auto min = std::numeric_limits<double>::max();
    //     int argmin = -1;
    //     for (int j = 0; j < block_width; ++j)
    //     {
    //         int global_j = block_j + i + j;
    //         const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
    //         if (distance < min and (global_j < block_i + i - exclude or global_j > block_i + i + exclude))
    //         {
    //             min = distance;
    //             argmin = global_j;
    //         }
    //     }
    //     EXPECT_NEAR(block_local_min_row[i].value, min, double(1e-14)) << "Values of the local min row are not equal to the explicit calculation at row " << i;
    //     EXPECT_EQ(block_local_min_row[i].index, argmin) << "Indices of the local min row are not equal to the explicit calculation";
    // }

}


// Test for STOMP_
TEST_F(BlockTest, QuandrangleWoInitialize)
{
    int n = 100;
    int m = 3;
    int exclude = 2;

    // Generate a random vector of doubles of size n
    std::vector<double> time_series;
    std::mt19937 gen(123); // Use a fixed seed of 123
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < n; i++)
    {
        time_series.push_back(dis(gen));
    }
    int block_width = 10;
    int block_height = 15;
    int block_i = 0;
    int block_j = -11;
    int block_ID = 0;

    std::vector<double> first_row(n - m + 1, 0.0);
    std::span view = std::span(&time_series[0], m);
    for (int j = 0; j < n - m + 1; ++j)
    {
        const auto distance = dotProduct(view, std::span(&time_series[j], m));
        first_row[j] = distance;
    }
    std::vector<double> initial_row(block_width, 0.0);

    // Initialize a Block object
    block<double> block(n-m+1, m, exclude, block_i, block_j, block_ID, block_width, block_height, first_row, initial_row, time_series);

    // Assert if the block type is correct
    ASSERT_EQ(block.get_type(), QUADRANGLE_WITHOUT_INITIAL_RECURRENCE) << "Block type is not 'quadrangle_without_initial_recurrence'";
    // Call the STOMP method
    block.STOMP();
    std::vector<double> block_row = block.get_row();
    std::vector<min_pair<double>> block_local_min_row = block.get_local_min_rows();

    for (int i = 0; i < block_width; ++i)
    {
        int global_i = block_i + i;
        view = std::span(&time_series[global_i], m);
        auto min = std::numeric_limits<double>::max();
        int argmin = -1;
        for (int j = 0; j < i - 1; ++j)
        {
            // int row_j = block_width - 1 - j;
            const auto distance = dotProduct(view, std::span(&time_series[j], m));
            if (distance < min and (j < block_i + i - exclude or j > block_i + i + exclude))
            {
                min = distance;
                argmin = j;
                // printf("row %d min: %f, argmin: %d\n",i,  min, argmin);
            }
        }

        EXPECT_NEAR(block_local_min_row[i].value, min, double(1e-14)) << "Values of the local min row are not equal to the explicit calculation at row " << i;
        EXPECT_EQ(block_local_min_row[i].index, argmin) << "Indices of the local min row are not equal to the explicit calculation";
    }
    for (int i = block_width; i < block_height; ++i)
    {
        view = std::span(&time_series[i], m);
        auto min = std::numeric_limits<double>::max();
        int argmin = -1;
        for (int j = 0; j < block_width; ++j)
        {
            int global_j = block_j + i + j;
            const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
            if (distance < min and (global_j < block_i + i - exclude or global_j > block_i + i + exclude))
            {
                min = distance;
                argmin = global_j;
            }
        }
        EXPECT_NEAR(block_local_min_row[i].value, min, double(1e-14)) << "Values of the local min row are not equal to the explicit calculation at row " << i;
        EXPECT_EQ(block_local_min_row[i].index, argmin) << "Indices of the local min row are not equal to the explicit calculation";
    }
}

TEST_F(BlockTest, QuandrangleWithInitialize)
{
    int n = 100;
    int m = 3;
    int exclude = 2;

    // Generate a random vector of doubles of size n
    std::vector<double> time_series;
    std::mt19937 gen(123); // Use a fixed seed of 123
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < n; i++)
    {
        time_series.push_back(dis(gen));
    }
    int block_width = 15;
    int block_height = 10;
    int block_i = 10;
    int block_j = -11;
    int block_ID = 0;

    std::vector<double> first_row(n - m + 1, 0.0);
    std::span view = std::span(&time_series[0], m);
    for (int j = 0; j < n - m + 1; ++j)
    {
        const auto distance = dotProduct(view, std::span(&time_series[j], m));
        first_row[j] = distance;
    }
    std::vector<double> initial_row(block_width, 0.0);
    for (int j = 0; j < block_width; ++j)
    {
        int global_j = block_j + j;
        if (global_j < 0 or j==block_width-1) {
        } else {
            initial_row[j+1] = dotProduct(std::span(&time_series[block_i-1], m), std::span(&time_series[block_j+j], m));
        }
    }
    for (int j = 0; j < block_width; ++j)
    {
        printf("%f ", initial_row[j]);
    }
    // Initialize a Block object
    block<double> block(n-m+1, m, exclude, block_i, block_j, block_ID, block_width, block_height, first_row, initial_row, time_series);

    // Assert if the block type is correct
    ASSERT_EQ(block.get_type(), QUADRANGLE_WITH_INITIAL_RECURRENCE) << "Block type is not 'quadrangle_with_initial_recurrence'";
    // Call the STOMP method
    block.STOMP();
    std::vector<double> block_row = block.get_row();
    std::vector<min_pair<double>> block_local_min_row = block.get_local_min_rows();

    view = std::span(&time_series[block_i + block_height - 1], m);
    for (int j = 2; j < block_width; ++j)
    {
        int global_j = block_j + block_height + j - 1;
        const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
        EXPECT_NEAR(block_row[j], distance, double(1e-14)) << "Elements of the row's final row are not equal to the explicit calculation";
    }

    int elem_per_row = 4;
    for (int i = 0; i <  block_height; ++i)
    {
        view = std::span(&time_series[block_i + i], m);
        auto min = std::numeric_limits<double>::max();
        int argmin = -1;
        for (int j = block_width - elem_per_row; j < block_width; ++j)
        {
            int global_j = block_j + i + j;
            const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
            if (distance < min and (global_j < block_i + i - exclude or global_j > block_i + i + exclude))
            {
                min = distance;
                argmin = global_j;
            }
        }
        EXPECT_NEAR(block_local_min_row[i].value, min, double(1e-14)) << "Values of the local min row are not equal to the explicit calculation at row " << i;
        EXPECT_EQ(block_local_min_row[i].index, argmin) << "Indices of the local min row are not equal to the explicit calculation";
        elem_per_row++;
    }
}

TEST_F(BlockTest, RightTruncatedParallelogram)
{
    int n = 102;
    int m = 3;
    int exclude = 2;

    // Generate a random vector of doubles of size n
    std::vector<double> time_series;
    std::mt19937 gen(123); // Use a fixed seed of 123
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < n; i++)
    {
        time_series.push_back(dis(gen));
    }
    int block_width = 10;
    int block_height = 10;
    int block_i = 0;
    int block_j = 90;
    int block_ID = 0;

    std::vector<double> first_row(n - m + 1);
    std::span view = std::span(&time_series[0], m);
    for (int j = block_j; j < n - m + 1; ++j)
    {
        const auto distance = dotProduct(view, std::span(&time_series[j], m));
        first_row[j] = distance;
    }

    std::vector<double> initial_row(block_width);
    for (int j = 0; j < block_width; ++j)
    {
        initial_row[j] = first_row[j];
    }
    // Initialize a Block object
    block<double> block(n-m+1, m, exclude, block_i, block_j, block_ID, block_width, block_height, first_row, initial_row, time_series);

    // Assert if the block type is correct
    ASSERT_EQ(block.get_type(), RIGHT_TRUNCATED_PARALLELOGRAM) << "Block type is not 'right_truncated_parallelogram'";

    // Call the STOMP method
    block.STOMP();

    std::vector<double> block_row = block.get_row();
    std::vector<min_pair<double>> block_local_min_row = block.get_local_min_rows();

    // Assert if the last row of the block is correct
    view = std::span(&time_series[block_height - 1], m);
    for (int j = 0; j < block_width; ++j)
    {
        int global_j = block_j + block_height + j - 1;
        if (global_j < n - m + 1)
        {
            const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
            EXPECT_DOUBLE_EQ(block_row[j], distance) << "Elements of the row's final row are not equal to the explicit calculation";
        }
    }
    // Assert if the local min row is correct
    for (int i = block_i; i < block_i+block_height; ++i)
    {
        view = std::span(&time_series[i], m);
        auto min = std::numeric_limits<double>::max();
        int argmin = -1;
        for (int j = 0; j < block_width; ++j)
        {
            int global_j = block_j + i + j;
            if (global_j < n - m + 1)
            {
                const auto distance = dotProduct(view, std::span(&time_series[global_j], m));
                if (distance < min and (global_j < block_i + i - exclude or global_j > block_i + i + exclude))
                {
                    min = distance;
                    argmin = global_j;
                }
            }
        }
        EXPECT_NEAR(block_local_min_row[i].value, min, double(1e-14)) << "Values of the local min row are not equal to the explicit calculation at row " << i;
        EXPECT_EQ(block_local_min_row[i].index, argmin) << "Indices of the local min row are not equal to the explicit calculation";
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}