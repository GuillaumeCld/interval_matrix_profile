#pragma once
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <omp.h>
#include "distance.hpp"
#include "block.hpp"
#include "vblock.hpp"

/**
 * @brief Compute the Interval Matrix Profile and index with the Block Interval Matrix Profile algorithm (BIMP)
 * 
 * @tparam array_value_t 
 * @tparam array_index_t 
 * @param time_series the time series
 * @param window_size the susbsequence length (m in the paper)
 * @param period_starts the starting index of each period (in order)
 * @param interval_length the length of the interval (L in the paper)
 * @param exclude the number of excluded points (e in the paper)
 * @return pair of matrix profile and profile index 
 */
template <typename array_value_t, typename array_index_t>
auto BIMP(array_value_t &time_series,
                                   const int window_size,
                                   array_index_t const &period_starts,
                                   const int interval_length,
                                   const int exclude) 
{
    using value_t = array_value_t::value_type;
    using index_t = array_index_t::value_type;
    using pair_t = min_pair<value_t>;
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int metarows = period_starts.size();
    const int half_interval = interval_length / 2;

    const int nb_blocks = (window_size > half_interval) ? metarows : metarows + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, -1);

    std::vector<value_t> first_row(n_sequence);
    int block_height;
    int block_width;
    int block_i;
    int block_j;

    #pragma omp parallel default(none)                                               \
    shared(time_series, matrix_profile, profile_index, first_row, period_starts) \
    firstprivate(n, n_sequence, metarows, interval_length, half_interval, nb_blocks, window_size, exclude) private(block_height, block_width, block_i, block_j)
    {
        std::span view = std::span(&time_series[0], window_size);
        #pragma omp single
        {
            // Compute the first row of the matrix profile
            #pragma omp taskloop
            for (int j = 0; j < n_sequence; ++j)
            {
                first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
            }
            #pragma omp taskwait
            // Iterate over the metarows
            #pragma omp taskloop untied
            for (int metarow = 0; metarow < metarows; ++metarow)
            {
                block_i = period_starts[metarow];
                // Compute the height of the current block
                block_height = (metarow == metarows - 1) ? n_sequence - period_starts[metarow] : period_starts[metarow + 1] - period_starts[metarow];
                // Initialize the local minimum per rows
                std::vector<pair_t> local_min_row(block_height, {-1, std::numeric_limits<value_t>::max()});
                // Iterate over the blocks
                for (int column = -1; column < nb_blocks; ++column)
                {
                    // Compute the width and j coordinate of the current block
                    if (column == -1)
                    {
                        block_width = interval_length + 1;
                        block_j = -block_height - half_interval;
                    }
                    else if (column < metarows)
                    {

                        block_width = (column == metarows - 1) ? std::min(interval_length + 1, n_sequence - (period_starts[column] - half_interval)) : interval_length + 1;
                        block_j = period_starts[column] - half_interval;
                    }
                    else
                    {
                        block_j = n - half_interval + 1;
                        if (block_j > n_sequence)
                        {
                            break;
                        }
                    }

                    // Create the block
                    block<value_t, false> block(n_sequence,
                                                window_size,
                                                exclude,
                                                block_i,
                                                block_j,
                                                column,
                                                block_width,
                                                block_height,
                                                first_row,
                                                time_series,
                                                local_min_row);
                    block.compute();
                    local_min_row = std::move(block.get_local_min_rows());
                }
                // Compute the global minimums per row and update the matrix profile/index
                for (int i = 0; i < block_height; ++i)
                {
                    matrix_profile[block_i + i] = std::sqrt(std::abs(local_min_row[i].value));
                    profile_index[block_i + i] = local_min_row[i].index;
                }
            }// end row taskloop
        }// end single
    }// end parallel region
    return std::make_pair(matrix_profile, profile_index);
}

/**
 * @brief Compute the Interval Matrix Profile and index with the Block Interval Matrix Profile algorithm (BIMP) vith vectorized block.
 * 
 * @tparam array_value_t 
 * @tparam array_index_t 
 * @param time_series the time series
 * @param window_size the susbsequence length (m in the paper)
 * @param period_starts the starting index of each period (in order)
 * @param interval_length the length of the interval (L in the paper)
 * @param exclude the number of excluded points (e in the paper)
 * @return pair of matrix profile and profile index 
 */
template <typename array_value_t, typename array_index_t>
auto vBIMP(array_value_t &time_series,
                                   const int window_size,
                                   array_index_t const &period_starts,
                                   const int interval_length,
                                   const int exclude)
{
    using value_t = array_value_t::value_type;
    using index_t = array_index_t::value_type;
    using pair_t = min_pair<value_t>;
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int metarows = period_starts.size();
    const int half_interval = interval_length / 2;

    const int nb_blocks = (window_size > half_interval) ? metarows : metarows + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, -1);

    std::vector<value_t> first_row(n_sequence);
    int block_height;
    int block_width;
    int block_i;
    int block_j;

    #pragma omp parallel default(none)                                               \
    shared(time_series, matrix_profile, profile_index, first_row, period_starts)     \
    firstprivate(n, n_sequence, metarows, interval_length, half_interval, nb_blocks, window_size, exclude) private(block_height, block_width, block_i, block_j)
    {
        std::span view = std::span(&time_series[0], window_size);
        #pragma omp single
        {
            // Compute the first row of the matrix profile
            #pragma omp taskloop
            for (int j = 0; j < n_sequence; ++j)
            {
                first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
            }
            #pragma omp taskwait
            // Iterate over the metarows
            #pragma omp taskloop untied
            for (int metarow = 0; metarow < metarows; ++metarow)
            {
                block_i = period_starts[metarow];
                // Compute the height of the current block
                block_height = (metarow == metarows - 1) ? n_sequence - period_starts[metarow] : period_starts[metarow + 1] - period_starts[metarow];
                // Initialize the local minimum per rows
                std::vector<pair_t> local_min_row(block_height, {-1, std::numeric_limits<value_t>::max()});
                // Iterate over the blocks
                for (int column = -1; column < nb_blocks; ++column)
                {
                    // Compute the width and j coordinate of the current block
                    if (column == -1)
                    {
                        block_width = interval_length + 1;
                        block_j = -block_height - half_interval;
                    }
                    else if (column < metarows)
                    {

                        block_width = (column == metarows - 1) ? std::min(interval_length + 1, n_sequence - (period_starts[column] - half_interval)) : interval_length + 1;
                        block_j = period_starts[column] - half_interval;
                    }
                    else
                    {
                        block_j = n - half_interval;
                        if (block_j > n_sequence)
                        {
                            break;
                        }
                    }

                    // Create the block
                    vblock<value_t, false> block(n_sequence,
                                                window_size,
                                                exclude,
                                                block_i,
                                                block_j,
                                                column,
                                                block_width,
                                                block_height,
                                                first_row,
                                                time_series,
                                                local_min_row);
                    block.compute();
                    local_min_row = std::move(block.get_local_min_rows());
                }
                // Compute the global minimums per row and update the matrix profile/index
                for (int i = 0; i < block_height; ++i)
                {
                    matrix_profile[block_i + i] = std::sqrt(std::abs(local_min_row[i].value));
                    profile_index[block_i + i] = local_min_row[i].index;
                }
            }// end row taskloop
        }// end single
    }// end parallel region
    return std::make_pair(matrix_profile, profile_index);
}

/**
 * @brief Compute the left Interval Matrix Profile and index with BIMP 
 * 
 * @tparam array_value_t 
 * @tparam array_index_t 
 * @param time_series 
 * @param window_size 
 * @param period_starts 
 * @param interval_length 
 * @param exclude 
 * @return auto 
 */
template <typename array_value_t, typename array_index_t>
auto left_BIMP(array_value_t &time_series,
                const int window_size,
                array_index_t const &period_starts,
                const int interval_length,
                const int exclude)
{
    using value_t = array_value_t::value_type;
    using index_t = array_index_t::value_type;
    using pair_t = min_pair<value_t>;
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int metarows = period_starts.size();
    const int half_interval = interval_length / 2;

    const int nb_blocks = (window_size > half_interval) ? metarows : metarows + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, -1);

    std::vector<value_t> first_row(n_sequence);
    int block_height;
    int block_width;
    int block_i;
    int block_j;

    #pragma omp parallel default(none)                                                                     \
    shared(time_series, matrix_profile, profile_index, first_row, period_starts)                           \
    firstprivate(n, n_sequence, metarows, interval_length, half_interval, nb_blocks, window_size, exclude) \
    private(block_height, block_width, block_i, block_j)
    {
        std::span view = std::span(&time_series[0], window_size);
        #pragma omp single
        {
            // Compute the first row of the matrix profile
            #pragma omp taskloop
            for (int j = 0; j < n_sequence; ++j)
            {
                first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
            }
            #pragma omp taskwait
            // Iterate over the metarows
            for (int metarow = 1; metarow < metarows; ++metarow)
            {
                #pragma omp task default(none)                                                                                                \
                    shared(first_row, time_series, n_sequence, window_size, n, metarows, interval_length, half_interval, nb_blocks, exclude)  \
                    firstprivate(metarow)                                                                                                     \
                    private(block_height, block_width, block_i, block_j)                                                                      \
                    untied                                                                                                                    \
                    priority(metarow)
                {
                    block_i = period_starts[metarow];
                    // Compute the height of the current block
                    block_height = (metarow == metarows - 1) ? n_sequence - period_starts[metarow] : period_starts[metarow + 1] - period_starts[metarow];
                    // Initialize the local minimum per rows
                    std::vector<pair_t> local_min_row(block_height, {-1, std::numeric_limits<value_t>::max()});
                    // Iterate over the blocks
                    for (int column = -1; column < metarow; ++column) // sub diagonal column only
                    {
                        // Compute the width and j coordinate of the current block
                        if (column == -1)
                        {
                            block_width = interval_length + 1;
                            block_j = -block_height - half_interval;
                        }
                        else 
                        {

                            block_width = (column == metarows - 1) ? std::min(interval_length + 1, n_sequence - (period_starts[column] - half_interval)) : interval_length + 1;
                            block_j = period_starts[column] - half_interval;
                        }

                        // Create the block
                        vblock<value_t, false> block(n_sequence,
                                                    window_size,
                                                    exclude,
                                                    block_i,
                                                    block_j,
                                                    column,
                                                    block_width,
                                                    block_height,
                                                    first_row,
                                                    time_series,
                                                    local_min_row);
                        block.compute();
                        local_min_row = std::move(block.get_local_min_rows());
                    }
                    // Compute the global minimums per row and update the matrix profile/index
                    for (int i = 0; i < block_height; ++i)
                    {
                        matrix_profile[block_i + i] = std::sqrt(std::abs(local_min_row[i].value));
                        profile_index[block_i + i] = local_min_row[i].index;
                    }
                }// end takse
            }// end row loop
        }// end single
    }// end parallel region
    return std::make_pair(matrix_profile, profile_index);
}

/**
 * @brief Compute the right Interval Matrix Profile and index with BIMP
 * 
 * @tparam array_value_t 
 * @tparam array_index_t 
 * @param time_series 
 * @param window_size 
 * @param period_starts 
 * @param interval_length 
 * @param exclude 
 * @return auto 
 */
template <typename array_value_t, typename array_index_t>
auto right_BIMP(array_value_t &time_series,
                const int window_size,
                array_index_t const &period_starts,
                const int interval_length,
                const int exclude)
{
    using value_t = array_value_t::value_type;
    using index_t = array_index_t::value_type;
    using pair_t = min_pair<value_t>;
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int metarows = period_starts.size();
    const int half_interval = interval_length / 2;

    const int nb_blocks = (window_size > half_interval) ? metarows : metarows + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, -1);

    std::vector<value_t> first_row(n_sequence);
    int block_height;
    int block_width;
    int block_i;
    int block_j;

    #pragma omp parallel default(none)                                                                     \
    shared(time_series, matrix_profile, profile_index, first_row, period_starts)                           \
    firstprivate(n, n_sequence, metarows, interval_length, half_interval, nb_blocks, window_size, exclude) \
    private(block_height, block_width, block_i, block_j)
    {
        std::span view = std::span(&time_series[0], window_size);
        #pragma omp single
        {
            // Compute the first row of the matrix profile
            #pragma omp taskloop
            for (int j = 0; j < n_sequence; ++j)
            {
                first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
            }
            #pragma omp taskwait
            // Iterate over the metarows
            for (int metarow = 0; metarow < metarows-1; ++metarow)
            {
                #pragma omp task default(none)                                                                                                \
                    shared(first_row, time_series, n_sequence, window_size, n, metarows, interval_length, half_interval, nb_blocks, exclude)  \
                    firstprivate(metarow)                                                                                                     \
                    private(block_height, block_width, block_i, block_j)                                                                      \
                    untied                                                                                                                    \
                    priority(metarows-metarow)
                {
                    block_i = period_starts[metarow];
                    // Compute the height of the current block
                    block_height = period_starts[metarow + 1] - period_starts[metarow];
                    // Initialize the local minimum per rows
                    std::vector<pair_t> local_min_row(block_height, {-1, std::numeric_limits<value_t>::max()});
                    // Iterate over the blocks
                    for (int column = metarow+1; column < nb_blocks; ++column) // over diagonal column only
                    {
                        // Compute the width and j coordinate of the current block
                        if (column < metarows)
                        {

                            block_width = (column == metarows - 1) ? std::min(interval_length + 1, n_sequence - (period_starts[column] - half_interval)) : interval_length + 1;
                            block_j = period_starts[column] - half_interval;
                        }
                        else
                        {
                            block_j = n - half_interval + 1;
                            if (block_j > n_sequence)
                            {
                                break;
                            }
                        }
                        // Create the block
                        vblock<value_t, false> block(n_sequence,
                                                    window_size,
                                                    exclude,
                                                    block_i,
                                                    block_j,
                                                    column,
                                                    block_width,
                                                    block_height,
                                                    first_row,
                                                    time_series,
                                                    local_min_row);
                        block.compute();
                        local_min_row = std::move(block.get_local_min_rows());
                    }
                    // Compute the global minimums per row and update the matrix profile/index
                    for (int i = 0; i < block_height; ++i)
                    {
                        matrix_profile[block_i + i] = std::sqrt(std::abs(local_min_row[i].value));
                        profile_index[block_i + i] = local_min_row[i].index;
                    }
                }// end takse
            }// end row loop
        }// end single
    }// end parallel region
    return std::make_pair(matrix_profile, profile_index);
}

template <typename array_value_t, typename array_index_t>
auto BIMP_intialized(array_value_t &time_series,
                                               const int window_size,
                                               array_index_t const &period_starts,
                                               const int interval_length,
                                               const int exclude)
{
    using value_t = array_value_t::value_type;
    using index_t = array_index_t::value_type;
    using pair_t = min_pair<value_t>;
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int metarows = period_starts.size();
    const int half_interval = interval_length / 2;

    const int nb_blocks = (window_size > half_interval) ? metarows : metarows + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, -1);

    std::vector<value_t> first_row(n_sequence);
    int block_height;
    int block_width;
    int block_i;
    int block_j;

    #pragma omp parallel default(none)                                               \
    shared(time_series, matrix_profile, profile_index, first_row, period_starts) \
    firstprivate(n, n_sequence, metarows, interval_length, half_interval, nb_blocks, window_size, exclude) private(block_height, block_width, block_i, block_j)
    {
        std::span view = std::span(&time_series[0], window_size);
        // Compute the first row of the matrix profile
        #pragma omp for
        for (int j = 0; j < n_sequence; ++j)
        {
            first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
        }
        #pragma omp barrier
        // Iterate over the metarows
        #pragma omp for
        for (int metarow = 0; metarow < metarows; ++metarow)
        {
            block_i = period_starts[metarow];
            // Compute the height of the current block
            block_height = (metarow == metarows - 1) ? n_sequence - period_starts[metarow] : period_starts[metarow + 1] - period_starts[metarow];
            // Initialize the local minimum per rows
            std::vector<pair_t> local_min_row(block_height, {-1, std::numeric_limits<value_t>::max()});
            // Iterate over the blocks
            for (int column = -1; column < nb_blocks; ++column)
            {
                // Compute the width of the current block
                if (column == -1)
                {
                    block_width = interval_length + 1;
                    block_j = -block_height - half_interval;
                }
                else if (column < metarows)
                {

                    block_width = (column == metarows - 1) ? std::min(interval_length + 1, n_sequence - (period_starts[column] - half_interval)) : interval_length + 1;
                    block_j = period_starts[column] - half_interval;
                }
                else
                {
                    block_j = n + 1 - half_interval;
                    if (block_j >= n_sequence)
                    {
                        break;
                    }
                }
                // Initialize the first row of the block
                std::span<value_t> initial_row;
                std::vector<value_t> tmp(block_width, value_t(0));
                if (column != -1)
                {
                    if (metarow == 0)
                    {
                        if (block_j < 0)
                        {
                            for (int j = half_interval - 1; j < block_width; ++j)
                            {
                                tmp[j] = first_row[j + 1 - half_interval];
                            }
                            initial_row = std::span(tmp.data(), block_width);
                        }
                        else
                        {
                            initial_row = std::span(&first_row[block_j], block_width);
                        }
                    }
                    else
                    {
                        std::span<value_t> view = std::span(&time_series[block_i - 1], window_size);
                        int start_index = (block_j < 0) ? half_interval : 0;
                        for (int j = start_index; j < block_width; ++j)
                        {
                            tmp[j] = dotProduct(view, std::span(&time_series[block_j + j - 1], window_size));
                        }
                        initial_row = std::span(tmp.data(), block_width);
                    }
                }

                // Create the block
                block<value_t> block(n_sequence,
                                     window_size,
                                     exclude,
                                     block_i,
                                     block_j,
                                     column,
                                     block_width,
                                     block_height,
                                     first_row,
                                     initial_row,
                                     time_series,
                                     local_min_row);
                block.compute();
                local_min_row = std::move(block.get_local_min_rows());
            }
            // Compute the global minimums per row and update the matrix profile/index
            for (int i = 0; i < block_height; ++i)
            {
                matrix_profile[block_i + i] = std::sqrt(std::abs(local_min_row[i].value));
                profile_index[block_i + i] = local_min_row[i].index;
            }
        }
    }
    return std::make_pair(matrix_profile, profile_index);
}