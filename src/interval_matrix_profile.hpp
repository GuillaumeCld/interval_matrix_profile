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
#include "vblock.hpp"

/**
 * @brief Interval Matrix Profile algorithm using the brute force procedure.
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
auto imp_bf(array_value_t &time_series,
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
            // if (place_in_period >= 365)
            // {
            //     place_in_period = place_in_period - 365;
            // }
            auto view = std::span(&time_series[i], window_size);
            auto min = std::numeric_limits<value_t>::max();
            int min_index = 0;
            const int i_pos = i - i_start;

            for (int j_period = 0; j_period < n_periods_i; ++j_period)
            {
                const int j_start = std::max(period_starts[j_period] + place_in_period - half_interval, 0);
                const int j_end = std::min(period_starts[j_period] + place_in_period + half_interval + 1, n_sequence);
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
                const int j_end = half_interval - (i_end - i)+1;
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

            if (is_periodic and i - i_start <= half_interval)
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




/**
 * @brief Interval Matrix Profile algorithm using the modified AAMP procedure, which computes the matrix profile then keeps the valid values.
 * 
 * @tparam array_value_t 
 * @tparam array_index_t 
 * @param time_series 
 * @param window_size 
 * @param period_starts 
 * @param interval_length 
 * @param exclude 
 * @param k 
 * @return auto 
 */
template <typename array_value_t, typename array_index_t>
auto modified_AAMP(array_value_t &time_series,
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


            if (i == 0)
            {
                for (int j = n_sequence - 1; j >= 0; --j)
                {
                    auto view = std::span(&time_series[i], window_size);
                    const auto distance = dotProduct(view, std::span(&time_series[j], window_size));
                    row_values[j] = distance;
                } 
            }

            for (int j = n_sequence - 1; j > 0; --j)
            {
                const auto prev_data{time_series[i - 1] - time_series[j - 1]};
                const auto next_data{time_series[i + window_size - 1] - time_series[j + window_size - 1]};
                const auto distance = row_values[j - 1] + (next_data * next_data - prev_data * prev_data);
                row_values[j] = distance;
            }

            auto view = std::span(&time_series[i], window_size);
            const auto distance = dotProduct(view, std::span(&time_series[0], window_size));
            row_values[0] = distance;
                
            for (int j_period = 0; j_period < n_periods_i; ++j_period)
            {
                const int j_start = std::max(period_starts[j_period] + i_pos - half_interval, 0);
                const int j_end = std::min(period_starts[j_period] + i_pos + half_interval + 1, n_sequence);
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
                const int j_end = half_interval - (i_end - i) + 1;
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





/** 
 * @brief Interval Matrix Profile algorithm using the brute force procedure with k nearest neighbors.
 * 
 * @tparam array_value_t 
 * @tparam array_index_t 
 * @param time_series 
 * @param window_size 
 * @param period_starts 
 * @param interval_length 
 * @param exclude 
 * @param k 
 * @return auto 
 */
template <typename array_value_t, typename array_index_t>
auto imp_bf_knn(array_value_t &time_series,
                                             const int window_size,
                                             array_index_t const &period_starts,
                                             const int interval_length,
                                             const int exclude,
                                             const int k)
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
        const int block_height = i_end - i_start;
        std::vector<heap_type> local_heap_row(block_height, heap_type(lesser));
        for (int i = i_start; i < i_end; ++i)
        {
            // create and fill the heap
            heap_type &heap = local_heap_row.at(i - i_start);

            for (int iter = 0; iter < k; ++iter)
            {
                pair_t tmp = {0, std::numeric_limits<value_t>::max()};
                heap.push(tmp);
            }

            // const int place_in_period = std::min(i - i_start, 365);
            int place_in_period = i - i_start;
            if (place_in_period > 365)
            {
                place_in_period = 1;
            }
            auto view = std::span(&time_series[i], window_size);

            const int i_pos = i - i_start;
            for (int j_period = 0; j_period < n_periods_i; ++j_period)
            {
                value_t min = std::numeric_limits<value_t>::max();
                index_t min_index = 0;
                const int j_start = std::max(period_starts[j_period] + place_in_period - half_interval, 0);
                const int j_end = std::min(period_starts[j_period] + place_in_period + half_interval + 1, n_sequence);
                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const value_t distance = dotProduct(view, std::span(&time_series[j], window_size));
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
                if (min < heap.top().value)
                {
                    // if (k==7 and i==13519 ) std::cout << "min " << min << " heap top " << heap.top().value << " Size " << heap.size() << std::endl;
                    heap.pop();
                    heap.push({ min_index, min});
                    // if (k==7 and i==13519 ) std::cout << "heap top " << heap.top().value << std::endl;
                }
            }

            if (i_end - i <= half_interval)
            {
                value_t min = std::numeric_limits<value_t>::max();
                index_t min_index = 0;
                const int j_start = 0;
                const int j_end = half_interval - (i_end - i ) + 1;
                for (int j = j_start; j < j_end; ++j)
                {
                    const value_t distance = dotProduct(view, std::span(&time_series[j], window_size));
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
                // if (k==7 and i==13519 ) std::cout << "2- min " << min << " j_start " << j_start << " j_end " << j_end << std::endl;
                if (min < heap.top().value)
                {
                    // if (k==7 and i==13519 ) std::cout << "2- min " << min << " heap top " << heap.top().value << " Size " << heap.size() << std::endl;
                    heap.pop();
                    heap.push({ min_index, min});
                    // if (k==7 and i==13519 ) std::cout << "2- heap top " << heap.top().value << std::endl;
                }
            }

            if (is_periodic and i - i_start < half_interval)
            {

                const int j_start = n + i_pos - half_interval;
                const int j_end = n_sequence;
                value_t min = std::numeric_limits<value_t>::max();
                index_t min_index = 0;
                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const value_t distance = dotProduct(view, std::span(&time_series[j], window_size));
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
                // if (k==7 and i==13519 ) std::cout << "3- min " << min << " j_start " << j_start << " j_end " << j_end << " " <<  heap.top().value  << std::endl;

                if (min < heap.top().value)
                {
                    // if (k==7 and i==13519 ) std::cout << "3- min " << min << " heap top " << heap.top().value << " Size " << heap.size() << std::endl;
                    heap.pop();
                    heap.push({ min_index, min});
                    // if (k==7 and i==13519 ) std::cout << "3- heap top " << heap.top().value << std::endl;
                }
                // 
            }
            pair_t min = heap.top();
            // if (k==7 and i==13519 ) std::cout << "Final min " << min.value << std::endl;
            matrix_profile[i] = std::sqrt(min.value);
            profile_index[i] = min.index;

        }
    }

    return std::make_pair(matrix_profile, profile_index);
}



/**
 * @brief Compute the z-normalized Interval Matrix Profile using the brute force procedure.
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
auto z_normalized_IMP_bf(array_value_t &time_series,
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
            int place_in_period = i - i_start;

            std::span<value_t> view = std::span(&time_series[i], window_size);
            auto min = std::numeric_limits<value_t>::max();
            int min_index = 0;
            const int i_pos = i - i_start;

            for (int j_period = 0; j_period < n_periods_i; ++j_period)
            {
                const int j_start = std::max(period_starts[j_period] + place_in_period - half_interval, 0);
                const int j_end = std::min(period_starts[j_period] + place_in_period + half_interval + 1, n_sequence);
                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const auto distance = znormalized_dotProduct(view, std::span(&time_series[j], window_size));
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
                const int j_end = half_interval - (i_end - i)+1;
                for (int j = j_start; j < j_end; ++j)
                {
                    const auto distance = znormalized_dotProduct(view, std::span(&time_series[j], window_size));
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
            }

            if (is_periodic and i - i_start <= half_interval)
            {

                const int j_start = n + i_pos - half_interval;
                const int j_end = n_sequence;

                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const auto distance = znormalized_dotProduct(view, std::span(&time_series[j], window_size));
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


/**
 * @brief Interval Matrix Profile algorithm using the brute force procedure with traditional k nearest neighbors.
 * 
 * @tparam array_value_t 
 * @tparam array_index_t 
 * @param time_series 
 * @param window_size 
 * @param period_starts 
 * @param interval_length 
 * @param exclude 
 * @param k 
 * @return auto 
 */
template <typename array_value_t, typename array_index_t>
auto interval_matrix_profile_brute_force_bad_NN(array_value_t &time_series,
                                         const int window_size,
                                         array_index_t const &period_starts,
                                         const int interval_length,
                                         const int exclude,
                                         const int k)
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
            heap_type heap(lesser);
            for (int iter = 0; iter < k; ++iter)
            {
                pair_t tmp = {0, std::numeric_limits<value_t>::max()};
                heap.push(tmp);
            }
            int place_in_period = i - i_start;

            auto view = std::span(&time_series[i], window_size);
            const int i_pos = i - i_start;

            for (int j_period = 0; j_period < n_periods_i; ++j_period)
            {
                const int j_start = std::max(period_starts[j_period] + place_in_period - half_interval, 0);
                const int j_end = std::min(period_starts[j_period] + place_in_period + half_interval + 1, n_sequence);
                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const auto distance = dotProduct(view, std::span(&time_series[j], window_size));
                    if (distance < heap.top().value and (j < i - exclude or j > i + exclude))
                    {
                        const pair_t min_pair = {j, distance};
                        heap.pop();
                        heap.push(min_pair);
                    }
                }
            }
            if (i_end - i <= half_interval)
            {
                const int j_start = 0;
                const int j_end = half_interval - (i_end - i)+1;
                for (int j = j_start; j < j_end; ++j)
                {
                    const auto distance = dotProduct(view, std::span(&time_series[j], window_size));
                    if (distance < heap.top().value  and (j < i - exclude or j > i + exclude))
                    {
                        const pair_t min_pair = {j, distance};
                        heap.pop();
                        heap.push(min_pair);
                    }
                }
            }
            if (is_periodic and i - i_start <= half_interval)
            {

                const int j_start = n + i_pos - half_interval;
                const int j_end = n_sequence;

                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const auto distance = dotProduct(view, std::span(&time_series[j], window_size));
                    if (distance < heap.top().value and (j < i - exclude or j > i + exclude))
                    {
                        const pair_t min_pair = {j, distance};
                        heap.pop();
                        heap.push(min_pair);
                    }
                }
            }
            const pair_t min_pair = heap.top();
            matrix_profile[i] = std::sqrt(min_pair.value);
            profile_index[i] = min_pair.index;
        }
    }

    return std::make_pair(matrix_profile, profile_index);
}