#pragma once
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <omp.h>
#include <queue>
#include <chrono>
#include "distance.hpp"
#include "block.hpp"
#include "block_kNN.hpp"
#include "k_nn_lookup.hpp"

template <typename array_value_t, typename array_index_t>
auto interval_matrix_profile_brute_force(array_value_t &time_series,
                                         const int window_size,
                                         array_index_t const &period_starts,
                                         const int interval_length,
                                         const int exclude)
{
    using value_t = array_value_t::value_type;
    using index_t = array_index_t::value_type;

    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, 0);
    const int half_interval = interval_length / 2;
    bool is_periodic = window_size <= half_interval;
    const int n_periods_i = period_starts.size();

    #pragma omp parallel for schedule(static)
    for (int i_period = 0; i_period < n_periods_i; ++i_period)
    {
        const int i_start = period_starts[i_period];
        const int i_end = (i_period == n_periods_i - 1) ? n_sequence : period_starts[i_period + 1];
        for (int i = i_start; i < i_end; ++i)
        {
            // const int place_in_period = std::min(i - i_start, 365);
            int place_in_period = i - i_start;
            if (place_in_period > 365)
            {
                place_in_period = 1;
            }
            auto view = std::span(&time_series[i], window_size);
            auto min = std::numeric_limits<value_t>::max();
            int min_index = 0;
            const int i_pos = i - i_start;

            for (int j_period = 0; j_period < n_periods_i; ++j_period)
            {
                const int j_start = std::max(period_starts[j_period] + place_in_period - half_interval, 0);
                const int j_end = std::min(period_starts[j_period] + place_in_period + half_interval+1, n_sequence);
                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const auto distance = dotProduct(view, std::span(&time_series[j], window_size));
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
            }

            if (i_end - i <= half_interval)
            {
                const int j_start = 0;
                const int j_end = half_interval - (i_end - i); 
                for (int j = j_start; j < j_end; ++j)
                {
                    const auto distance = dotProduct(view, std::span(&time_series[j], window_size));
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
            }

            if (is_periodic and i - i_start < half_interval)
            {

                const int j_start = n + i_pos - half_interval;
                const int j_end = n_sequence;
                
                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const auto distance = dotProduct(view, std::span(&time_series[j], window_size));
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
            }
            matrix_profile[i] = std::sqrt(min);
            profile_index[i] = min_index;
        }
    }

    return std::make_pair(matrix_profile, profile_index);
}

template <typename array_value_t, typename array_index_t>
auto interval_matrix_profile_STOMP(array_value_t &time_series,
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

    const int nb_blocks = (window_size > half_interval) ? metarows + 1 : metarows + 2;
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
                    block_j = - block_height - half_interval;
                }
                else if (column < metarows)
                {

                    block_width = (column == metarows -1) ? std::min(interval_length+1, n_sequence - (period_starts[column] - half_interval)) : interval_length+1;
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
                block.STOMP();
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

template <typename array_value_t, typename array_index_t>
auto interval_matrix_profile_STOMP_initialized(array_value_t &time_series,
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

    const int nb_blocks = (window_size > half_interval) ? metarows + 1: metarows + 2;
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
                    block_j = - block_height - half_interval;
                }
                else if (column < metarows)
                {

                    block_width = (column == metarows - 1) ? std::min(interval_length+1, n_sequence - (period_starts[column] - half_interval)) : interval_length+1;
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
                block.STOMP();
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


template <typename array_value_t, typename array_index_t>
auto interval_matrix_profile_STOMP_kNN(array_value_t &time_series,
                                       const int window_size,
                                       array_index_t const &period_starts,
                                       const int interval_length,
                                       const int exclude,
                                       const int k)

{

    using value_t = array_value_t::value_type;
    using index_t = array_index_t::value_type;
    using pair_t = min_pair<value_t>;
    using container_t = std::vector<pair_t>;
    auto lesser = [](pair_t const &a, pair_t const &b) -> bool
    { return a.value < b.value; };
    using heap_type = typename std::priority_queue<pair_t, container_t, decltype(lesser)>;

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

#pragma omp parallel default(none)                                                                  \
    shared(time_series, matrix_profile, profile_index, first_row, period_starts, lesser, std::cout) \
    firstprivate(k, n, n_sequence, metarows, interval_length, half_interval, nb_blocks, window_size, exclude) private(block_height, block_width, block_i, block_j)
    {
        auto cumulative_time = 0.0;
        auto cumulative_time_heaps = 0.0;
        auto cumulative_time_stomp = 0.0;
        auto cumulative_time_block = 0.0;
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
            // auto start_init = std::chrono::high_resolution_clock::now();
            std::vector<heap_type> local_heap_row(block_height, heap_type(lesser));
            // auto end_init = std::chrono::high_resolution_clock::now();
            // auto duration_init = std::chrono::duration_cast<std::chrono::microseconds>(end_init - start_init).count();
            // cumulative_time_heaps += duration_init;
            // Iterate over the blocks
            for (int column = 0; column < nb_blocks; ++column)
            {
                // Compute the width of the current block
                if (column < metarows)
                {

                    block_width = (column == metarows - 1) ? std::min(interval_length, n_sequence - (period_starts[column] - half_interval)) : interval_length;
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
                    // std::cout << "Start index " << start_index << std::endl << std::flush;
                    for (int j = start_index; j < block_width; ++j)
                    {
                        tmp[j] = dotProduct(view, std::span(&time_series[block_j + j - 1], window_size));
                    }
                    initial_row = std::span(tmp.data(), block_width);
                }

                // Create the block
                // auto start_block = std::chrono::high_resolution_clock::now();
                block_kNN<value_t, heap_type> block(n_sequence,
                                              window_size,
                                              exclude,
                                              block_i,
                                              block_j,
                                              column,
                                              block_width,
                                              block_height,
                                              k,
                                              first_row,
                                              initial_row,
                                              time_series,
                                              &local_heap_row);
                // auto end_block = std::chrono::high_resolution_clock::now();
                // auto duration_block = std::chrono::duration_cast<std::chrono::microseconds>(end_block - start_block).count();
                // cumulative_time_block += duration_block;

                // auto start_stomp = std::chrono::high_resolution_clock::now();
                block.STOMP();
                // auto end_stomp = std::chrono::high_resolution_clock::now();
                // auto duration_stomp = std::chrono::duration_cast<std::chrono::microseconds>(end_stomp - start_stomp).count();
                // cumulative_time_stomp += duration_stomp;
                // local_heap_row = std::move(block.get_heap_per_row());
            }
            // Compute the global minimums per row and update the matrix profile/index
            // auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < block_height; ++i)
            {
                auto k_nn = extract_k_min_from_heap<value_t, heap_type, pair_t>(local_heap_row.at(i), k, exclude);
                const int heap_size = local_heap_row.at(i).size();
                std::vector<value_t> k_nn_values = k_nn.first;
                std::vector<index_t> k_nn_indices = k_nn.second;
                // std::cout << "Sizes " << heap_size << " " << k_nn_values.size() << " " << k_nn_indices.size() << " k: " << k << std::endl;
                matrix_profile[block_i + i] = std::sqrt(std::abs(k_nn_values.at(k - 1)));
                profile_index[block_i + i] = k_nn_indices.at(k - 1);
            }
            // auto end = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            // cumulative_time += duration;
        }
        // std::cout << "Cumulative time heap init  " << cumulative_time_heaps / 1000.0 << " ms" << std::endl;
        // std::cout << "Cumulative time heap processing " << cumulative_time / 1000.0 << " ms" << std::endl;
        // std::cout << "Cumulative time block creation " << cumulative_time_block / 1000.0 << " ms" << std::endl;
        // std::cout << "Cumulative time stomp processing " << cumulative_time_stomp / 1000.0 << " ms" << std::endl;
    }

    return std::make_pair(matrix_profile, profile_index);
}



template <typename array_value_t, typename array_index_t>
auto modified_STOMP(array_value_t &time_series,
                                         const int window_size,
                                         array_index_t const &period_starts,
                                         const int interval_length,
                                         const int exclude)
{
    using value_t = array_value_t::value_type;
    using index_t = array_index_t::value_type;

    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, 0);
    const int half_interval = interval_length / 2;
    bool is_periodic = window_size <= half_interval;
    const int n_periods_i = period_starts.size();

    std::vector<value_t> row_values(n_sequence, 0);

    for (int i_period = 0; i_period < n_periods_i; ++i_period)
    {
        const int i_start = period_starts[i_period];
        const int i_end = (i_period == n_periods_i - 1) ? n_sequence : period_starts[i_period + 1];
        for (int i = i_start; i < i_end; ++i)
        {
            auto min = std::numeric_limits<value_t>::max();
            int min_index = 0;
            const int i_pos = i - i_start;
            for (int j = n_sequence-1; j >= 0; --j)
            {
                // Compute Euclidean distance for the current sequence
                if (i == 0 or j == 0)
                {
                    auto view = std::span(&time_series[i], window_size);
                    const auto distance = dotProduct(view, std::span(&time_series[j], window_size));
                    row_values[j] = distance;
                }
                else
                {
                    const auto prev_data{time_series[i - 1] - time_series[j - 1]};
                    const auto next_data{time_series[i + window_size - 1] - time_series[j + window_size - 1]};
                    const auto distance = row_values[j-1] + (next_data * next_data - prev_data * prev_data);
                    row_values[j] = distance;

                }

            }
            for (int j_period = 0; j_period < n_periods_i; ++j_period)
                {
                    const int j_start = std::max(period_starts[j_period] + i_pos - half_interval, 0);
                    const int j_end = std::min(period_starts[j_period] + i_pos + half_interval+1, n_sequence);
                    for (int j = j_start; j < j_end; ++j)
                    {
                        // Compute Euclidean distance for the current sequence
                        const auto distance = row_values[j];
                        if (distance < min and (j < i - exclude or j > i + exclude))
                        {
                            min = distance;
                            min_index = j;
                        }
                    }
                }

                if (i_end - i <= half_interval)
                {
                    const int j_start = 0;
                    const int j_end = half_interval - (i_end - i); 
                    for (int j = j_start; j < j_end; ++j)
                    {
                        const auto distance = row_values[j];
                        if (distance < min and (j < i - exclude or j > i + exclude))
                        {
                            min = distance;
                            min_index = j;
                        }
                    }
                }
                if (is_periodic and i - i_start < half_interval)
                {

                    const int j_start = n + i_pos - half_interval;
                    const int j_end = n_sequence;
                    
                    for (int j = j_start; j < j_end; ++j)
                    {
                        // Compute Euclidean distance for the current sequence
                        const auto distance = row_values[j];
                        if (distance < min and (j < i - exclude or j > i + exclude))
                        {
                            min = distance;
                            min_index = j;
                        }
                    }
                }
                    


            matrix_profile[i] = std::sqrt(std::abs(min));
            profile_index[i] = min_index;
        }
    }

    return std::make_pair(matrix_profile, profile_index);
}





template <typename array_value_t, typename array_index_t>
auto BIMP_kNN(array_value_t &time_series,
                const int window_size,
                array_index_t const &period_starts,
                const int interval_length,
                const int exclude,
                const int k,
                const bool exclude_diagonal)
{
    using value_t = array_value_t::value_type;
    using index_t = array_index_t::value_type;
    using pair_t = min_pair<value_t>;
    // Heap type
    using container_t = std::vector<pair_t>;
    auto lesser = [](pair_t const &a, pair_t const &b) -> bool
    { return a.value < b.value; };
    using heap_type = typename std::priority_queue<pair_t, container_t, decltype(lesser)>;


    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int metarows = period_starts.size();
    const int half_interval = interval_length / 2;

    const int nb_blocks = (window_size > half_interval) ? metarows + 1 : metarows + 2;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, -1);

    std::vector<value_t> first_row(n_sequence);
    int block_height;
    int block_width;
    int block_i;
    int block_j;

    #pragma omp parallel default(none)                                               \
        shared(time_series, matrix_profile, profile_index, first_row, period_starts, lesser) \
        firstprivate(k, n, n_sequence, metarows, interval_length, half_interval, nb_blocks, window_size, exclude, exclude_diagonal) private(block_height, block_width, block_i, block_j)
    {
        std::span view = std::span(&time_series[0], window_size);
        // Compute the first row of the matrix profile
        #pragma omp for
        for (int j = 0; j < n_sequence; ++j)
        {
            first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
        }
        // Iterate over the metarows
        #pragma omp taskloop untied
        for (int metarow = 0; metarow < metarows; ++metarow)
        {
            block_i = period_starts[metarow];
            // Compute the height of the current block
            block_height = (metarow == metarows - 1) ? n_sequence - period_starts[metarow] : period_starts[metarow + 1] - period_starts[metarow];
            // Initialize the local heaps per rows
            std::vector<heap_type> local_heap_row(block_height, heap_type(lesser));
            // Iterate over the blocks
            for (int column = -1; column < nb_blocks; ++column)
            {

                if (exclude_diagonal and column == metarow)
                {
                    continue;
                }
                // Compute the width of the current block
                if (column == -1)
                {
                    block_width = interval_length + 1;
                    block_j = - block_height - half_interval;
                }
                else if (column < metarows)
                {

                    block_width = (column == metarows -1) ? std::min(interval_length+1, n_sequence - (period_starts[column] - half_interval)) : interval_length+1;
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
                std::vector<pair_t> local_min_row(block_height, {-1, std::numeric_limits<value_t>::max()});
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
                block.STOMP();
                auto const & min_row = block.get_local_min_rows();
                for (int i = 0; i < block_height; ++i)
                {
                    const auto min_pair = min_row.at(i);
                    heap_type& local_heap = local_heap_row.at(i);
                    if (local_heap.size() < k)
                    {
                        local_heap.push(min_pair);
                    }
                    else
                    {
                        if (min_pair.value < local_heap.top().value)
                        {
                            local_heap.pop();
                            local_heap.push(min_pair);
                        }
                    }
                }
            }
            // Compute the global minimums per row and update the matrix profile/index
            for (int i = 0; i < block_height; ++i)
            {
                auto min_pair = local_heap_row.at(i).top();
                matrix_profile[block_i + i] = std::sqrt(std::abs(min_pair.value));
                profile_index[block_i + i] = min_pair.index;
            }
        }
    } // end parallel
    return std::make_pair(matrix_profile, profile_index);
}
