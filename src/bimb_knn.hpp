#pragma once
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <omp.h>
#include <queue>
#include "distance.hpp"
#include "vblock.hpp"


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

    const int nb_blocks = (window_size > half_interval) ? metarows  : metarows + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, -1);

    std::vector<value_t> first_row(n_sequence);
    int block_height;
    int block_width;
    int block_i;
    int block_j;

    #pragma omp parallel default(none)                                                                                                                                                   \
        shared(time_series, matrix_profile, profile_index, first_row, period_starts, lesser, std::cout)                                                                                  \
        firstprivate(k, n, n_sequence, metarows, interval_length, half_interval, nb_blocks, window_size, exclude, exclude_diagonal) private(block_height, block_width, block_i, block_j)
    {
        std::span view = std::span(&time_series[0], window_size);
        // Compute the first row of the matrix profile
        #pragma omp for
        for (int j = 0; j < n_sequence; ++j)
        {
            first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
        }


        #pragma omp single
        {
        // Iterate over the metarows
            #pragma omp taskloop untied
            for (int metarow = 0; metarow < metarows; ++metarow)
            {
                block_i = period_starts[metarow];
                // Compute the height of the current block
                block_height = (metarow == metarows - 1) ? n_sequence - period_starts[metarow] : period_starts[metarow + 1] - period_starts[metarow];
                // Initialize the local heaps per rows
                std::vector<heap_type> local_heap_row(block_height, heap_type(lesser));
                for (int i = 0; i < block_height; ++i)
                {
                    for (int iter = 0; iter < k; ++iter)
                    {
                        pair_t tmp = {0, std::numeric_limits<value_t>::max()};
                        local_heap_row.at(i).push(tmp);
                    }
                }
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
                        // std::cout << "Block j " << block_j << " n_sequence " << n_sequence << " block i " << block_i << " block height " << block_height << std::endl;
                        if (block_j >= n_sequence)
                        {
                            break;
                        }
                    }
                    std::vector<pair_t> local_min_row(block_height, {-1, std::numeric_limits<value_t>::max()});
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
                    auto const &min_row = block.get_local_min_rows();
                    
                    for (int i = 0; i < block_height; ++i)
                    {
                        const auto min_pair = min_row.at(i);
                        heap_type &local_heap = local_heap_row.at(i);
                        if (min_pair.value < local_heap.top().value)
                        {
                            local_heap.pop();
                            local_heap.push(min_pair);
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
            }// end row loop
        }// end single
    } // end parallel
    return std::make_pair(matrix_profile, profile_index);
}

template <typename array_value_t, typename array_index_t>
auto left_BIMP_kNN(array_value_t &time_series,
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

    const int nb_blocks = (window_size > half_interval) ? metarows  : metarows + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, -1);

    std::vector<value_t> first_row(n_sequence);
    int block_height;
    int block_width;
    int block_i;
    int block_j;

    #pragma omp parallel default(none)                                                                                              \
        shared(time_series, matrix_profile, profile_index, first_row, period_starts, lesser)                             \
        firstprivate(k, n, n_sequence, metarows, interval_length, half_interval, nb_blocks, window_size, exclude, exclude_diagonal) \
        private(block_height, block_width, block_i, block_j)
    {
        std::span view = std::span(&time_series[0], window_size);
        // Compute the first row of the matrix profile
        #pragma omp for
        for (int j = 0; j < n_sequence; ++j)
        {
            first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
        }
        // Iterate over the metarows
        #pragma single
        {
            for (int metarow = 1; metarow < metarows; ++metarow)
            {

                #pragma omp task default(none)                                                                                                \
                    shared(time_series, first_row, matrix_profile, profile_index, k, n_sequence, window_size, n, metarows, interval_length, half_interval, nb_blocks, exclude, lesser, period_starts, exclude_diagonal)  \
                    firstprivate(metarow)                                                                                                     \
                    private(block_height, block_width, block_i, block_j)                                                                      \
                    untied                                                                                                                    \
                    priority(metarow)
                {
                    block_i = period_starts[metarow];
                    // Compute the height of the current block
                    block_height = (metarow == metarows - 1) ? n_sequence - period_starts[metarow] : period_starts[metarow + 1] - period_starts[metarow];
                    // Initialize the local heaps per rows
                    std::vector<heap_type> local_heap_row(block_height, heap_type(lesser));
                    for (int i = 0; i < block_height; ++i)
                    {
                        for (int iter = 0; iter < k; ++iter)
                        {
                            pair_t tmp = {0, std::numeric_limits<value_t>::max()};
                            local_heap_row.at(i).push(tmp);
                        }
                    }
                    // Iterate over the blocks
                    for (int column = -1; column < metarow; ++column)
                    {

                        if (exclude_diagonal and column == metarow)
                        {
                            continue;
                        }
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
                        std::vector<pair_t> local_min_row(block_height, {-1, std::numeric_limits<value_t>::max()});
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
                        auto const &min_row = block.get_local_min_rows();
                        
                        for (int i = 0; i < block_height; ++i)
                        {
                            const auto min_pair = min_row.at(i);
                            heap_type &local_heap = local_heap_row.at(i);
                            if (min_pair.value < local_heap.top().value)
                            {
                                local_heap.pop();
                                local_heap.push(min_pair);
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
                }// end task
            }// end row loop
        }// end single
    } // end parallel
    return std::make_pair(matrix_profile, profile_index);
}

template <typename array_value_t, typename array_index_t>
auto right_BIMP_kNN(array_value_t &time_series,
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

    const int nb_blocks = (window_size > half_interval) ? metarows  : metarows + 1;
    std::vector<value_t> matrix_profile(n_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence, -1);

    std::vector<value_t> first_row(n_sequence);
    int block_height;
    int block_width;
    int block_i;
    int block_j;

    #pragma omp parallel default(none)                                                       \
        shared(time_series, matrix_profile, profile_index, first_row, period_starts, lesser, std::cout) \
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
        #pragma omp  single
        {
            for (int metarow = 0; metarow < metarows-1; ++metarow)
            {
                #pragma omp task default(none)                                                                                                \
                    shared(time_series, first_row, matrix_profile, profile_index, k, n_sequence, window_size, n, metarows, interval_length, half_interval, nb_blocks, exclude, lesser, period_starts, exclude_diagonal)  \
                    firstprivate(metarow)                                                                                                     \
                    private(block_height, block_width, block_i, block_j)                                                                      \
                    untied                                                                                                                    \
                    priority(metarows-metarow)
                {
                    block_i = period_starts[metarow];
                    // Compute the height of the current block
                    block_height = (metarow == metarows - 1) ? n_sequence - period_starts[metarow] : period_starts[metarow + 1] - period_starts[metarow];
                    // Initialize the local heaps per rows
                    std::vector<heap_type> local_heap_row(block_height, heap_type(lesser));
                    for (int i = 0; i < block_height; ++i)
                    {
                        for (int iter = 0; iter < k; ++iter)
                        {
                            pair_t tmp = {0, std::numeric_limits<value_t>::max()};
                            local_heap_row.at(i).push(tmp);
                        }
                    }
                    // Iterate over the blocks
                    for (int column = metarow+1; column < nb_blocks; ++column)
                    {

                        if (exclude_diagonal and column == metarow)
                        {
                            continue;
                        }
                        // Compute the width of the current block
                        if (column < metarows)
                        {

                            block_width = (column == metarows - 1) ? std::min(interval_length + 1, n_sequence - (period_starts[column] - half_interval)) : interval_length + 1;
                            block_j = period_starts[column] - half_interval;
                        }
                        else
                        {
                            block_j = n - half_interval;
                            if (block_j >= n_sequence)
                            {
                                break;
                            }
                        }
                        std::vector<pair_t> local_min_row(block_height, {-1, std::numeric_limits<value_t>::max()});
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
                        auto const &min_row = block.get_local_min_rows();
                        
                        for (int i = 0; i < block_height; ++i)
                        {
                            const auto min_pair = min_row.at(i);
                            heap_type &local_heap = local_heap_row.at(i);
                            if (min_pair.value < local_heap.top().value)
                            {
                                local_heap.pop();
                                local_heap.push(min_pair);
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
                }// end task
            }// end row loop
        }// end single
    } // end parallel
    return std::make_pair(matrix_profile, profile_index);
}