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
template <typename T>
auto computeMatrixProfileBruteForce(const std::vector<T> &data,
                                    const int window_size,
                                    const int exclude)
{
    int n_sequence = data.size() - window_size + 1;
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

#pragma omp parallel for shared(matrix_profile, profile_index, data) schedule(static)
    for (int i = 0; i < n_sequence; ++i)
    {
        std::span view = std::span(&data[i], window_size);
        auto min = std::numeric_limits<T>::max();
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

template <typename T>
auto brute_force_v2(const std::vector<T> &data,
                    int window_size)
{
    int n_sequence = data.size() - window_size + 1;
    int exclude = 2; // std::ceil(window_size / 2);
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

#pragma omp parallel for shared(matrix_profile, profile_index, data) schedule(static)
    for (int i = 0; i < n_sequence; ++i)
    {
        std::span view = std::span(&data[i], window_size);
        auto min = std::numeric_limits<T>::max();
        int min_index = 0;
        for (int j = 0; j < i - exclude; ++j)
        {
            const auto distance = dotProduct(view, std::span(&data[j], window_size));
            if (distance < min)
            {
                min = distance;
                min_index = j;
            }
        }
        for (int j = i + exclude; j < n_sequence; ++j)
        {
            const auto distance = dotProduct(view, std::span(&data[j], window_size));
            if (distance < min)
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
 * @brief Computes the matrix profile of a given time series data using the STOMP procedure.
 *
 * @tparam T The data type of the time series.
 * @param data The input time series data.
 * @param window_size The size of the window for computing the matrix profile.
 * @return std::pair<std::vector<T>,std::vector<int>> pair with the matrix profile and its index.
 */
template <typename array_t>
auto matrix_profile_STOMP(array_t &time_series, const int window_size, const int block_width, const int block_height, const int exclude) -> std::pair<std::vector<typename array_t::value_type>,std::vector<int>>
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



/**
 * @brief
 *
 * @tparam T
 * @param data
 * @param window_size
 * @return std::tuple<std::vector<T>,std::vector<int>> tuple with the matrix profile and its index.
 */
template <typename T>
auto computeMatrixProfileSTOMP(const std::vector<T> &data,
                               int window_size)
{
    int n_sequence = data.size() - window_size + 1;
    int exclude = 2; // std::ceil(window_size / 4);
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

    std::vector<T> first_row(n_sequence, T(0));
    std::vector<T> row(n_sequence, T(0));
    std::vector<T> oldrow(n_sequence, T(0));
    std::vector<T> nrow(n_sequence, T(0));

    int max_threads_num = omp_get_max_threads();
    std::vector<T> local_min_values(max_threads_num, std::numeric_limits<T>::max());
    std::vector<int> local_mins_indices(max_threads_num, 0);

#pragma omp parallel shared(first_row, row, nrow, matrix_profile, profile_index, local_min_values, local_mins_indices)
    {
        int threads_num = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int chunk_size = n_sequence / threads_num;
        int start = thread_id * chunk_size;
        int end = (thread_id == threads_num - 1) ? n_sequence : (thread_id + 1) * chunk_size;

        // printf("Parallel block with %d threads and chunks of size %d \n", threads_num, n_sequence / threads_num);

        // First row computed by chunk
        std::span view = std::span(&data[0], window_size);
        for (int j = start; j < end; ++j)
        {
            const auto distance = dotProduct(view, std::span(&data[j], window_size));
            first_row[j] = distance;
            row[j] = distance;

            // local minimum per chunk
            if (row[j] < local_min_values[thread_id] and (j < -exclude or j > exclude))
            {
                local_min_values[thread_id] = row[j];
                local_mins_indices[thread_id] = j;
            }
        }

        #pragma omp barrier
        #pragma omp single
        {
            // global minimum
            auto min = matrix_profile[0];
            int ind = 0;
            for (int k = 0; k < threads_num; ++k)
            {
                if (local_min_values[k] < min)
                {
                    min = local_min_values[k];
                    ind = local_mins_indices[k];
                }
                local_min_values[k] = std::numeric_limits<T>::max();
            }
            matrix_profile[0] = std::sqrt(min);
            profile_index[0] = ind;
        }
#pragma omp barrier
        // Compute all other rows
        for (int i = 1; i < n_sequence; ++i)
        {
            // Compute the i-th row of the distance matrix based on the (i-1)-th and the first one (first_row)
            // First chunk
            if (thread_id == 0)
            {
                nrow[0] = first_row[i];
                start = 1;
            }

            // Diagonal recurrence for all chunks
            for (int j = start; j < end; ++j)
            {
                nrow[j] = row[j - 1] - std::pow(data[j - 1] - data[i - 1], 2) + std::pow(data[j + window_size - 1] - data[i + window_size - 1], 2);
                // local minimum per chunk
                if (nrow[j] < local_min_values[thread_id] and (j < i - exclude or j > i + exclude))
                {
                    local_min_values[thread_id] = nrow[j];
                    local_mins_indices[thread_id] = j;
                }
            }
#pragma omp barrier
#pragma omp single
            {
                // Global minimum on row
                auto min = matrix_profile[i];
                int ind = 0;
                local_min_values[0] = std::min(local_min_values[0], first_row[i]);
                for (int k = 0; k < threads_num; ++k)
                {
                    if (local_min_values[k] < min)
                    {
                        min = local_min_values[k];
                        ind = local_mins_indices[k];
                    }
                    local_min_values[k] = std::numeric_limits<T>::max();
                }
                matrix_profile[i] = std::sqrt(std::abs(min));
                profile_index[i] = ind;
                auto tmp = row;
                row = nrow;
                nrow = oldrow;
                oldrow = tmp;
            }
#pragma omp barrier
        }
    }
    return std::make_pair(matrix_profile, profile_index);
}

template <typename T>
auto blockSTOMP(std::vector<T> &time_series, int window_size, int block_width, int block_height)
{
    // Time series parameters
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int exclude = 2;
    const int first_block_height = block_height;
    // Block parameters
    int n_blocks = n_sequence / block_width;
    if (n_sequence % block_width > 0)
    {
        n_blocks++;
    }
    int nb_metarows = n_sequence / block_height;
    if (n_sequence % block_height > 0)
    {
        nb_metarows++;
    }

    const int missing = static_cast<int>(std::ceil((n_sequence - (n_blocks - static_cast<int>(std::ceil(static_cast<T>(block_height) / block_width))) * block_width) / static_cast<T>(block_width)));
    n_blocks = n_blocks + missing;
    int n_total_blocks = n_blocks;
    // Initialize the matrix profile and its index
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, -1);
    // Array of local minimums per row for each block
    std::vector<std::vector<min_pair<T>>> block_min_pair_per_row(n_total_blocks + 2);
    // Compute the first row of the distance matrixmake
    std::vector<T> first_row(n_sequence);
    std::span view = std::span(&time_series[0], window_size);
    for (int j = 0; j < n_sequence; ++j)
    {
        const auto distance = dotProduct(view, std::span(&time_series[j], window_size));
        first_row[j] = distance;
    }
    //
    std::vector<block<T>> blocks(n_total_blocks + 2);
    // Loop to build the blocks
    int previous_diagonal_shift = block_height / block_width;
    if (block_height % block_width > 0)
    {
        previous_diagonal_shift++;
    }
    std::vector<T> initial_row(block_width, 0.0);
    for (int metarow = 0; metarow < nb_metarows; ++metarow)
    {
        int block_shift = (metarow + 1) * block_height;
        int diagonal_shift = block_shift / block_width;

        if (block_shift % block_width > 0)
        {
            diagonal_shift++;
        }
        int row_length = -diagonal_shift * block_width + n_blocks * block_width + metarow * block_height;
        if (row_length < n_sequence)
        {
            n_total_blocks = n_blocks + 1;
        }
        else
        {
            n_total_blocks = n_blocks;
        }
        if (metarow == nb_metarows - 1)
        {
            block_height = n_sequence - metarow * first_block_height;
        }
        for (int block_id = n_total_blocks - 1; block_id >= 0; --block_id)
        {
            // printf("Block %d\n", block_id);
            int block_j = -diagonal_shift * block_width + metarow * first_block_height + block_id * block_width;
            int block_i = metarow * first_block_height;
            // Retrieve the initial row for the block's recurrence
            if (metarow == 0)
            {
                for (int j = 0; j < block_width; ++j)
                {
                    initial_row[j] = first_row[block_j + j];
                }
            }
            else
            {
                int block_ind = std::max(0, block_id - (diagonal_shift - previous_diagonal_shift));
                initial_row = blocks[block_ind].get_row();
            }
            // Initialize the block

            block<T> block(n_sequence,
                           window_size,
                           exclude,
                           block_i,
                           block_j,
                           block_id,
                           block_width,
                           block_height,
                           first_row,
                           initial_row,
                           time_series);

            blocks[block_id] = std::move(block);
            blocks[block_id].compute();
            // retrieve the minimums per row
            block_min_pair_per_row[block_id] = blocks[block_id].get_local_min_rows();
        }
        // Compute the global minimums per row
        // printf("Compute global minimums\n   ");
        for (int i = 0; i < block_height; ++i)
        {
            auto min = std::numeric_limits<T>::max();
            int ind = -1;
            for (int k = 0; k < n_total_blocks; ++k)
            {
                auto min_pair = block_min_pair_per_row[k][i];

                if (min_pair.value < min)
                {
                    min = min_pair.value;
                    ind = min_pair.index;
                }
            }
            matrix_profile[metarow * first_block_height + i] = std::sqrt(std::abs(min));
            profile_index[metarow * first_block_height + i] = ind;
        }
        previous_diagonal_shift = diagonal_shift;
    }
    return std::make_pair(matrix_profile, profile_index);
}

template <typename T>
auto blockSTOMP_v2(std::vector<T> &time_series, const int window_size, const int block_width, const int block_height, const int exclude)
{
    // Time series parameters
    const int n = time_series.size();
    const int n_sequence{n - window_size + 1};
    const int first_block_height{block_height};
    int current_block_height{block_height};
    // Block parameters
    int n_blocks = (n_sequence % block_width == 0) ? n_sequence / block_width : n_sequence / block_width + 1;
    int nb_metarows = (n_sequence % block_height == 0) ? n_sequence / block_height : n_sequence / block_height + 1;
    const int missing = static_cast<int>(std::ceil((n_sequence - (n_blocks - static_cast<int>(std::ceil(static_cast<T>(block_height) / block_width))) * block_width) / static_cast<T>(block_width)));
    const int first_n_blocks = n_blocks + missing;
    int n_total_blocks = first_n_blocks;
    // Initialize the matrix profile and its index
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, -1);
    // Array of local minimums per row for each block
    std::vector<std::vector<min_pair<T>>> block_min_pair_per_row(n_total_blocks * 2);
    // Compute the first row of the distance matrixmake
    std::vector<T> first_row(n_sequence);
    std::span view = std::span(&time_series[0], window_size);

    std::vector<block<T>> current_blocks(n_total_blocks * 2);
    std::vector<std::vector<T>> previous_blocks(n_total_blocks * 2);
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

                        std::span<T> initial_row;
                        std::vector<T> tmp;
                        // Retrieve the initial row for the block's recurrence
                        if (metarow == 0)
                        {
                            if (block_j < 0)
                            {
                                tmp = std::vector<T>(block_width, 0.0);
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
                        block<T> block(n_sequence,
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
                    T min{std::numeric_limits<T>::max()};
                    int ind{-1};
                    for (int k = 0; k < n_total_blocks; ++k)
                    {
                        const min_pair<T> min_pair = block_min_pair_per_row[k][i];
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

template <typename T>
auto blockSTOMP_v3(std::vector<T> &time_series, int window_size, const int block_width, int block_height)
{
    // Time series parameters
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int exclude = 2;
    const int first_block_height = block_height;
    // Block parameters
    int n_blocks = (n_sequence % block_width == 0) ? n_sequence / block_width : n_sequence / block_width + 1;
    int nb_metarows = (n_sequence % block_height == 0) ? n_sequence / block_height : n_sequence / block_height + 1;
    const int missing = static_cast<int>(std::ceil((n_sequence - (n_blocks - static_cast<int>(std::ceil(static_cast<T>(block_height) / block_width))) * block_width) / static_cast<T>(block_width)));
    const int first_n_blocks = n_blocks + missing;
    int n_total_blocks = first_n_blocks;
    // Initialize the matrix profile and its index
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, -1);
    // Array of local minimums per row for each block
    std::vector<std::vector<min_pair<T>>> block_min_pair_per_row(n_total_blocks + 2);
    // Compute the first row of the distance matrixmake
    std::vector<T> first_row(n_sequence);
    std::span view = std::span(&time_series[0], window_size);

    //
    std::vector<block<T>> current_blocks(n_total_blocks + 2);
    std::vector<block<T>> previous_blocks(n_total_blocks + 2);
    // Loop to build the blocks
    int previous_diagonal_shift = (block_height % block_width > 0) ? block_height / block_width + 1 : block_height / block_width;
    min_pair<T> minimum = {-1, std::numeric_limits<T>::max()};
#pragma omp parallel shared(first_row, time_series, current_blocks, block_min_pair_per_row, view, first_n_blocks, \
                                n_sequence, window_size, first_block_height, block_width, previous_blocks, previous_diagonal_shift, minimum)
    {
#pragma omp for
        for (int j = 0; j < n_sequence; ++j)
        {
            first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
        }

        for (int metarow = 0; metarow < nb_metarows; ++metarow)
        {
            const int block_shift = (metarow + 1) * block_height;
            const int diagonal_shift = (block_shift % block_width > 0) ? block_shift / block_width + 1 : block_shift / block_width;
#pragma omp single
            {
                const int row_length = -diagonal_shift * block_width + first_n_blocks * block_width + metarow * block_height;
                n_total_blocks = (row_length < n_sequence) ? first_n_blocks + 1 : first_n_blocks;
                block_height = (metarow == nb_metarows - 1) ? n_sequence - metarow * first_block_height : first_block_height;
                for (int block_id = n_total_blocks - 1; block_id >= 0; --block_id)
                {
#pragma omp task default(none)                                                                                                                        \
    shared(first_row, time_series, current_blocks, block_min_pair_per_row, n_sequence, window_size, first_block_height, block_width, previous_blocks) \
    firstprivate(metarow, block_id, block_shift, diagonal_shift, previous_diagonal_shift, row_length, n_total_blocks, block_height)                   \
    untied
                    {
                        int previous_block_id = std::max(0, block_id - (diagonal_shift - previous_diagonal_shift));
                        int block_j = -diagonal_shift * block_width + metarow * first_block_height + block_id * block_width;
                        int block_i = metarow * first_block_height;
                        std::vector<T> initial_row(block_width, 0.0);

                        // Retrieve the initial row for the block's recurrence
                        if (metarow == 0)
                        {
                            for (int j = 0; j < block_width; ++j)
                            {
                                initial_row[j] = first_row[block_j + j];
                            }
                        }
                        else
                        {
                            initial_row = previous_blocks[previous_block_id].get_row();
                        }
                        // Initialize the block
                        block<T> block(n_sequence,
                                       window_size,
                                       exclude,
                                       block_i,
                                       block_j,
                                       block_id,
                                       block_width,
                                       block_height,
                                       first_row,
                                       initial_row,
                                       time_series);

                        current_blocks[block_id] = std::move(block);
                        current_blocks[block_id].compute();
                        // retrieve the minimums per row
                        block_min_pair_per_row[block_id] = current_blocks[block_id].get_local_min_rows();
                    }
                }
#pragma omp taskwait
            }
            // Compute the global minimums per row
            for (int i = 0; i < block_height; ++i)
            {
                // minimum = {-1, std::numeric_limits<T>::max()};
#pragma omp for reduction(min_pair_min : minimum)
                for (int k = 0; k < n_total_blocks; ++k)
                {
                    auto &min_pair = block_min_pair_per_row[k][i];
                    minimum = min_pair_min2(minimum, min_pair);
                }
#pragma omp single
                {
                    matrix_profile[metarow * first_block_height + i] = std::sqrt(std::abs(minimum.value));
                    profile_index[metarow * first_block_height + i] = minimum.index;
                    minimum = {-1, std::numeric_limits<T>::max()};
                }
                // #pragma omp barrier
            }
#pragma omp single
            {
                previous_diagonal_shift = diagonal_shift;
                previous_blocks = current_blocks;
            }
        }
    }
    return std::make_pair(matrix_profile, profile_index);
}

template <typename T>
auto blockSTOMP_v4(std::vector<T> &time_series, int window_size, const int block_width, int block_height)
{
    // Time series parameters
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int exclude = 2;
    const int first_block_height = block_height;
    // Block parameters
    int n_blocks = (n_sequence % block_width == 0) ? n_sequence / block_width : n_sequence / block_width + 1;
    int nb_metarows = (n_sequence % block_height == 0) ? n_sequence / block_height : n_sequence / block_height + 1;
    const int missing = static_cast<int>(std::ceil((n_sequence - (n_blocks - static_cast<int>(std::ceil(static_cast<T>(block_height) / block_width))) * block_width) / static_cast<T>(block_width)));
    const int first_n_blocks = n_blocks + missing;
    int n_total_blocks = first_n_blocks;
    // Initialize the matrix profile and its index
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, -1);
    // Array of local minimums per row for each block
    std::vector<std::vector<min_pair<T>>> block_min_pair_per_row(n_total_blocks + 2);
    // Compute the first row of the distance matrixmake
    std::vector<T> first_row(n_sequence);
    std::span view = std::span(&time_series[0], window_size);

    //
    std::vector<block<T>> current_blocks(n_total_blocks + 2);
    std::vector<std::vector<T>> previous_blocks(n_total_blocks + 2);
    // Loop to build the blocks
    int previous_diagonal_shift = (block_height % block_width > 0) ? block_height / block_width + 1 : block_height / block_width;
    min_pair<T> minimum = {-1, std::numeric_limits<T>::max()};
#pragma omp parallel shared(first_row, time_series, current_blocks, block_min_pair_per_row, view, first_n_blocks, \
                                n_sequence, window_size, first_block_height, block_width, previous_blocks, previous_diagonal_shift, minimum)
    {
#pragma omp for schedule(static)
        for (int j = 0; j < n_sequence; ++j)
        {
            first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
        }

        for (int metarow = 0; metarow < nb_metarows; ++metarow)
        {
            const int block_shift = (metarow + 1) * block_height;
            const int diagonal_shift = (block_shift % block_width > 0) ? block_shift / block_width + 1 : block_shift / block_width;
#pragma omp single
            {
                const int row_length = -diagonal_shift * block_width + first_n_blocks * block_width + metarow * block_height;
                n_total_blocks = (row_length < n_sequence) ? first_n_blocks + 1 : first_n_blocks;
                block_height = (metarow == nb_metarows - 1) ? n_sequence - metarow * first_block_height : first_block_height;
                for (int block_id = n_total_blocks - 1; block_id >= 0; --block_id)
                {
#pragma omp task default(none)                                                                                                                        \
    shared(first_row, time_series, current_blocks, block_min_pair_per_row, n_sequence, window_size, first_block_height, block_width, previous_blocks) \
    firstprivate(metarow, block_id, block_shift, diagonal_shift, previous_diagonal_shift, row_length, n_total_blocks, block_height)                   \
    untied
                    {
                        int previous_block_id = std::max(0, block_id - (diagonal_shift - previous_diagonal_shift));
                        int block_j = -diagonal_shift * block_width + metarow * first_block_height + block_id * block_width;
                        int block_i = metarow * first_block_height;
                        std::vector<T> initial_row(block_width, 0.0);
                        // Retrieve the initial row for the block's recurrence
                        if (metarow == 0) [[unlikely]]
                        {
                            for (int j = 0; j < block_width; ++j)
                            {
                                initial_row[j] = first_row[block_j + j];
                            }
                        }
                        else [[likely]]
                        {
                            initial_row = previous_blocks[previous_block_id]; //.get_row();
                        }
                        // Initialize the block
                        block<T> block(n_sequence,
                                       window_size,
                                       exclude,
                                       block_i,
                                       block_j,
                                       block_id,
                                       block_width,
                                       block_height,
                                       first_row,
                                       initial_row,
                                       time_series);

                        current_blocks[block_id] = std::move(block);
                        current_blocks[block_id].compute();
                        // retrieve the minimums per row
                        block_min_pair_per_row[block_id] = current_blocks[block_id].get_local_min_rows();
                    }
                }
#pragma omp taskwait
            }
            // Compute the global minimums per row
            for (int i = 0; i < block_height; ++i)
            {
#pragma omp for reduction(min_pair_min : minimum)
                for (int k = 0; k < n_total_blocks; ++k)
                {
                    auto &min_pair = block_min_pair_per_row[k][i];
                    minimum = min_pair_min2(minimum, min_pair);
                }
#pragma omp single
                {
                    matrix_profile[metarow * first_block_height + i] = std::sqrt(std::abs(minimum.value));
                    profile_index[metarow * first_block_height + i] = minimum.index;
                    minimum = {-1, std::numeric_limits<T>::max()};
                }
            }
#pragma omp single
            {
                previous_diagonal_shift = diagonal_shift;
                // previous_blocks = current_blocks;
                for (int block_id = n_total_blocks - 1; block_id >= 0; --block_id)
                {
                    previous_blocks[block_id] = std::move(current_blocks[block_id].get_row());
                }
            }
        }
    }
    return std::make_pair(matrix_profile, profile_index);
}

template <typename T>
auto blockSTOMP_v5(std::vector<T> &time_series, int window_size, const int block_width, int block_height)
{
    // Time series parameters
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int exclude = 2;
    const int first_block_height = block_height;
    // Block parameters
    int n_blocks = (n_sequence % block_width == 0) ? n_sequence / block_width : n_sequence / block_width + 1;
    int nb_metarows = (n_sequence % block_height == 0) ? n_sequence / block_height : n_sequence / block_height + 1;
    const int missing = static_cast<int>(std::ceil((n_sequence - (n_blocks - static_cast<int>(std::ceil(static_cast<T>(block_height) / block_width))) * block_width) / static_cast<T>(block_width)));
    const int first_n_blocks = n_blocks + missing;
    int n_total_blocks = first_n_blocks;
    // Initialize the matrix profile and its index
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, -1);
    // Array of local minimums per row for each block
    std::vector<std::vector<min_pair<T>>> block_min_pair_per_row(n_total_blocks + 2);
    // Compute the first row of the distance matrixmake
    std::vector<T> first_row(n_sequence);
    std::span view = std::span(&time_series[0], window_size);

    //
    std::vector<block<T>> current_blocks(n_total_blocks + 2);
    std::vector<std::vector<T>> previous_blocks(n_total_blocks + 2);
    // Loop to build the blocks
    int previous_diagonal_shift = (block_height % block_width > 0) ? block_height / block_width + 1 : block_height / block_width;
    min_pair<T> minimum = {-1, std::numeric_limits<T>::max()};
#pragma omp parallel shared(first_row, time_series, current_blocks, block_min_pair_per_row, view, first_n_blocks, \
                                n_sequence, window_size, first_block_height, block_width, previous_blocks, previous_diagonal_shift, minimum)
    {
#pragma omp for schedule(static)
        for (int j = 0; j < n_sequence; ++j)
        {
            first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
        }

        for (int metarow = 0; metarow < nb_metarows; ++metarow)
        {

            const int block_shift = (metarow + 1) * first_block_height;
            int block_i = metarow * first_block_height;

            const int diagonal_shift = (block_shift % block_width > 0) ? block_shift / block_width + 1 : block_shift / block_width;
#pragma omp single
            {
                const int row_length = -diagonal_shift * block_width + first_n_blocks * block_width + block_i;
                n_total_blocks = (row_length < n_sequence) ? first_n_blocks + 1 : first_n_blocks;
                block_height = (metarow == nb_metarows - 1) ? n_sequence - block_i : first_block_height;

                for (int block_id = n_total_blocks - 1; block_id >= 0; --block_id)
                {
#pragma omp task default(none)                                                                                                                        \
    shared(first_row, time_series, current_blocks, block_min_pair_per_row, n_sequence, window_size, first_block_height, block_width, previous_blocks) \
    firstprivate(metarow, block_id, block_shift, diagonal_shift, previous_diagonal_shift, row_length, n_total_blocks, block_height, block_i)          \
    untied
                    {
                        int previous_block_id{std::max(0, block_id - (diagonal_shift - previous_diagonal_shift))};
                        int block_j{block_width * (block_id - diagonal_shift) + block_i};

                        std::span<T> initial_row;
                        // Retrieve the initial row for the block's recurrence
                        if (metarow == 0) [[unlikely]]
                        {
                            initial_row = std::span(&first_row[block_j], block_width);
                        }
                        else [[likely]]
                        {
                            // auto &tmp = previous_blocks[previous_block_id];
                            initial_row = std::span(previous_blocks[previous_block_id].data(), block_width); //.get_row();
                        }
                        // Initialize the block
                        block<T> block(n_sequence,
                                       window_size,
                                       exclude,
                                       block_i,
                                       block_j,
                                       block_id,
                                       block_width,
                                       block_height,
                                       first_row,
                                       initial_row,
                                       time_series);
                        current_blocks[block_id] = std::move(block);
                        current_blocks[block_id].compute();
                        // retrieve the minimums per row
                        block_min_pair_per_row[block_id] = current_blocks[block_id].get_local_min_rows();
                    }
                }
            #pragma omp taskwait
            }
            // Compute the global minimums per row
            for (int i = 0; i < block_height; ++i)
            {
                #pragma omp for reduction(min_pair_min : minimum)
                for (int k = 0; k < n_total_blocks; ++k)
                {
                    auto &min_pair = block_min_pair_per_row[k][i];
                    minimum = min_pair_min2(minimum, min_pair);
                }
                #pragma omp single
                {
                    matrix_profile[metarow * first_block_height + i] = std::sqrt(std::abs(minimum.value));
                    profile_index[metarow * first_block_height + i] = minimum.index;
                    minimum = {-1, std::numeric_limits<T>::max()};
                }
            }
            #pragma omp single
            {
                previous_diagonal_shift = diagonal_shift;
                // previous_blocks = current_blocks;
                for (int block_id = n_total_blocks - 1; block_id >= 0; --block_id)
                {
                    previous_blocks[block_id] = std::move(current_blocks[block_id].get_row());
                }
            }
        }
    }
    return std::make_pair(matrix_profile, profile_index);
}


