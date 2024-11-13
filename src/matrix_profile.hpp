#pragma once
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <omp.h>
#include "distance.hpp"
#include "block.hpp"

/**
 * @brief Computes the matrix profile of a given time series data using the brute force procedure.
 *
 * @tparam T The data type of the time series.
 * @param data The input time series data.
 * @param window_size The size of the window for computing the matrix profile.
 * @return std::tuple<std::vector<T>,std::vector<int>> tuple with the matrix profile and its index.
 */
template <typename array_t>
auto MP_bf(const array_t &data,
            const int window_size,
            const int exclude)
{
    using value_t = array_t::value_type;

    int n_sequence = data.size() - window_size + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<int> profile_index(n_sequence, 0);

    #pragma omp parallel for shared(matrix_profile, profile_index, data) schedule(static)
    for (int i = 0; i < n_sequence; ++i)
    {
        std::span view = std::span(&data[i], window_size);
        auto min = std::numeric_limits<value_t>::max();
        int min_index = 0;
        for (int j = 0; j < n_sequence; ++j)
        {
            // Compute Euclidean distance for the current sequence
            const auto distance = dotProduct(view, std::span(&data[j], window_size));
            if (distance < min and (j < i - exclude or j > i + exclude))
            {
                min = distance;
                min_index = j;
            }
        }
        matrix_profile[i] = std::sqrt(min);
        profile_index[i] = min_index;
    }
    return std::make_pair(matrix_profile, profile_index);
}

/**
 * @brief Computes the matrix profile of a given time series data using the Block AAMP algorithm.
 *
 * @tparam T The data type of the time series.
 * @param data The input time series data.
 * @param window_size The size of the window for computing the matrix profile.
 * @return std::tuple<std::vector<T>,std::vector<int>> tuple with the matrix profile and its index.
 */
template <typename array_t>
auto BAAMP(array_t &time_series, const int window_size, const int block_width, const int block_height, const int exclude) -> std::pair<std::vector<typename array_t::value_type>,std::vector<int>>
{
    using value_t = array_t::value_type;
    // Time series parameters
    const int n = time_series.size();
    const int n_sequence{n - window_size + 1};
    const int first_block_height{block_height};
    int current_block_height{block_height};
    // Block parameters
    int n_blocks = (n_sequence % block_width == 0) ? n_sequence / block_width : n_sequence / block_width + 1;
    int nb_metarows = (n_sequence % block_height == 0) ? n_sequence / block_height : n_sequence / block_height + 1;
    const int missing = static_cast<int>(std::ceil((n_sequence - (n_blocks - static_cast<int>(std::ceil(static_cast<value_t>(block_height) / block_width))) * block_width) / static_cast<value_t>(block_width)));
    const int first_n_blocks = n_blocks + missing;
    int n_total_blocks = first_n_blocks;
    // Initialize the matrix profile and its index
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<int> profile_index(n_sequence, -1);
    // Array of local minimums per row for each block
    std::vector<std::vector<min_pair<value_t>>> block_min_pair_per_row(n_total_blocks * 2);
    // Compute the first row of the distance matrixmake
    std::vector<value_t> first_row(n_sequence);
    std::span view = std::span(&time_series[0], window_size);

    std::vector<block<value_t>> current_blocks(n_total_blocks * 2);
    std::vector<std::vector<value_t>> previous_blocks(n_total_blocks * 2);
    // Loop to build the blocks
    int previous_diagonal_shift = (block_height % block_width > 0) ? block_height / block_width + 1 : block_height / block_width;
    #pragma omp parallel shared(first_row, time_series, current_blocks, block_min_pair_per_row, \
                                n_sequence, window_size, first_block_height, block_width, exclude, previous_blocks, previous_diagonal_shift, view)
    {
        #pragma omp for schedule(static)
        for (int j = 0; j < n_sequence; ++j)
        {
            first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
        }
        #pragma omp single
        {
            for (int metarow = 0; metarow < nb_metarows; ++metarow)
            {
                const int block_i{metarow * first_block_height};
                const int block_shift{block_i + first_block_height};

                const int diagonal_shift = (block_shift % block_width > 0) ? block_shift / block_width + 1 : block_shift / block_width;
                const int row_length{(first_n_blocks - diagonal_shift) * block_width + block_i};

                n_total_blocks = (row_length < n_sequence) ? n_total_blocks + 1 : first_n_blocks;
                current_block_height = (metarow == nb_metarows - 1) ? n_sequence - block_i : first_block_height;
                for (int block_id = n_total_blocks - 1; block_id >= 0; --block_id)
                {
                    const int previous_block_id{std::max(0, block_id - (diagonal_shift - previous_diagonal_shift))};
                    const int block_j{(block_id - diagonal_shift) * block_width + block_i};

                    #pragma omp task default(none)                                                                                                                                                   \
                        shared(first_row, time_series, current_blocks, block_min_pair_per_row, n_sequence, window_size, first_block_height, block_width, exclude, previous_blocks, std::cout)                 \
                        firstprivate(metarow, block_id, block_shift, diagonal_shift, previous_diagonal_shift, row_length, n_total_blocks, current_block_height, block_i, previous_block_id, block_j) \
                        untied
                    // depend(in: previous_blocks[previous_block_id])
                    {

                        std::span<value_t> initial_row;
                        std::vector<value_t> tmp;
                        // Retrieve the initial row for the block's recurrence
                        if (metarow == 0)
                        {
                            if (block_j < 0)
                            {
                                tmp = std::vector<value_t>(block_width, 0.0);
                                const int n_valid = block_width + block_j;
                                if (n_valid > 0)
                                {
                                    const int start = block_width - n_valid;
                                   for (int j = 0; j < n_valid; ++j)
                                   {
                                       tmp[start + j] = first_row[j];
                                   }
                                }
                                initial_row = std::span(tmp.data(), block_width);
                            }
                            else
                            {
                                initial_row = std::span(&first_row.at(block_j), block_width);
                            }
                        }
                        else
                        {
                            initial_row = std::span(previous_blocks.at(previous_block_id).data(), block_width);
                        }
                        // Initialize the block
                        block<value_t> block(n_sequence,
                                       window_size,
                                       exclude,
                                       block_i,
                                       block_j,
                                       block_id,
                                       block_width,
                                       current_block_height,
                                       first_row,
                                       initial_row,
                                       time_series);
                        // Store the block
                        current_blocks.at(block_id) = std::move(block);
                        // Compute the block
                        current_blocks[block_id].compute();
                        // retrieve the minimums per row
                        block_min_pair_per_row.at(block_id) = current_blocks.at(block_id).get_local_min_rows();
                    }
                }
                #pragma omp taskwait
                // Compute the global minimums per row
                for (int i = 0; i < current_block_height; ++i)
                {
                    value_t min{std::numeric_limits<value_t>::max()};
                    int ind{-1};
                    for (int k = 0; k < n_total_blocks; ++k)
                    {
                        const min_pair<value_t> min_pair = block_min_pair_per_row[k][i];
                        if (min_pair.value < min)
                        {
                            min = min_pair.value;
                            ind = min_pair.index;
                        }
                    }

                    matrix_profile[block_i + i] = std::sqrt(std::abs(min));
                    profile_index[block_i + i] = ind;
                }
                previous_diagonal_shift = diagonal_shift;
                for (int block_id = 0; block_id < n_total_blocks; ++block_id)
                {
                    previous_blocks[block_id] = std::move(current_blocks[block_id].get_row());
                }
            }
        }
    }
    return std::make_pair(matrix_profile, profile_index);
}