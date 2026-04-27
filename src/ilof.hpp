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
 * @brief Compute the Interval Local Outlier Factor using the distance computation of the Block Interval Matrix Profile algorithm (BIMP) vith vectorized block.
 *
 * @tparam array_value_t
 * @tparam array_index_t
 * @param time_series
 * @param window_size
 * @param period_starts
 * @param interval_length
 * @param exclude
 * @param k
 * @param exclude_diagonal
 * @return auto
 */
template <typename array_value_t, typename array_index_t>
auto BILOF(array_value_t &time_series,
              const int window_size,
              array_index_t const &period_starts,
              const int interval_length,
              const int exclude,
              const int k,
              const bool exclude_diagonal)
{
    assert (window_size > 0 && "Window size must be greater than 0");
    assert (k > 0 && "k must be greater than 0");
    assert (interval_length > 0 && "Interval length must be greater than 0");


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

    const int nb_blocks = (window_size > half_interval) ? metarows : metarows + 1;
    std::vector<std::vector<pair_t>> kNNs(n_sequence, std::vector<pair_t>(k, {0, std::numeric_limits<value_t>::max()}));

    std::vector<value_t> first_row(n_sequence);
    int block_height;
    int block_width;
    int block_i;
    int block_j;

#pragma omp parallel default(none)                                         \
    shared(time_series, kNNs, first_row, period_starts, lesser, std::cout) \
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
                // Write the kNNs for the current metarow
                for (int i = 0; i < block_height; ++i)
                {
                    auto &local_heap = local_heap_row.at(i);
                    for (int iter = 0; iter < k; ++iter)
                    {
                        if (local_heap.empty())
                        {
                            break;
                        }
                        auto const &top = local_heap.top();
                        if (top.value < std::numeric_limits<value_t>::max())
                        {
                            kNNs[block_i + i][iter] = top;
                        }
                        local_heap.pop();
                    }
                }
            } // end row loop
        } // end single
    } // end parallel

    // Compute the Local Outlier Factor from the kNNs
    std::vector<value_t> lof(n_sequence, 0);
    std::vector<value_t> lrd(n_sequence, 0);

    // Compute the Local Reachability Density (LRD)
    for (int i = 0; i < n_sequence; ++i)
    {
        value_t sum_lrd{0};
        value_t d_ij;
        value_t dk_j;
        value_t reach_dist;

        for (int j = 0; j < k; ++j)
        {
            d_ij = kNNs[i][j].value;
            dk_j = kNNs[j][k - 1].value; // k-th nearest neighbor distance of the j-th point
            reach_dist = std::max(d_ij, dk_j);

            sum_lrd += reach_dist;
        }
        lrd[i] = (k > 0) ? k / sum_lrd : 0; // Local Reachability Density
    }
    // Compute the Local Outlier Factor (LOF)
    for (int i = 0; i < n_sequence; ++i)
    {
        value_t sum_lof{0};
        auto const &kNN = kNNs[i];
        for (int j = 0; j < k; ++j)
        {
            sum_lof += lrd[kNN[j].index] / lrd[i];
        }
        lof[i] = sum_lof / k; // Local Outlier Factor
    }
    return lof;
}