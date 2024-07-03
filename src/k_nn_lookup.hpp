#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <limits>
#include <queue>

/**
 * @brief Check if a value is inside the exclusion zone
 *
 * @param k_index, the index of the k values
 * @param i, the index of the value to check
 * @param exclude_zone, the size of the exclusion zone
 * @return bool, true if the value is inside the exclusion zone
 */
auto inline inside_exclusion_zone(std::vector<int> &k_index, const int i, const int exclude_zone) -> bool
{
    for (int j = 0; j < k_index.size(); ++j)
    {
        if (std::abs(i - k_index[j]) < exclude_zone)
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief Compute the k linear sweeps of a time series
 *
 * @tparam T
 * @param time_series
 * @param k
 * @param exclude_zone
 * @return auto
 */
template <typename T>
auto k_linear_sweeps(std::vector<T> &time_series, const int k, const int exclude_zone)
{

    std::vector<T> k_min;
    k_min.reserve(k);
    std::vector<int> k_index;
    k_index.reserve(k);

    for (int sweep = 0; sweep < k; ++sweep)
    {
        T min = std::numeric_limits<T>::max();
        int min_index = -1;
        for (int i = 0; i < time_series.size(); ++i)
        {
            if (time_series[i] < min and not inside_exclusion_zone(k_index, i, exclude_zone))
            {
                min = time_series[i];
                min_index = i;
            }
        }
        k_min.push_back(min);
        k_index.push_back(min_index);
    }
    return std::make_pair(k_min, k_index);
}

template <typename Pair_type>
auto inline my_greater(Pair_type const &a, Pair_type const &b) -> bool
{
    return a.first > b.first;
}

template <typename T>
auto k_nn_heap(std::vector<T> &time_series, const int k, const int exclude_zone)
{
    const int n = time_series.size();
    const int bound = k * exclude_zone;

    // Find the k*exclude_zone smallest values
    using value_t = std::pair<T, int>;
    using container_t = std::vector<value_t>;
    auto lesser = [](value_t const &a, value_t const &b) -> bool
    { return a.first < b.first; };
    std::priority_queue<value_t, container_t, decltype(lesser)> max_heap(lesser);

    for (int i = 0; i < bound; ++i)
    {
        max_heap.push(std::pair<T, int>(time_series[i], i));
    }
    for (int i = bound; i < n; ++i)
    {
        if (time_series[i] < max_heap.top().first)
        {
            max_heap.pop();
            max_heap.push(value_t(time_series[i], i));
        }
    }
    // Sort the values
    std::vector<value_t> sorted_values(bound);
    for (int i = 0; i < bound; ++i)
    {
        sorted_values[bound-1-i] = max_heap.top();
        max_heap.pop();
    }

    // Find the k smallest values
    std::vector<T> k_min;
    k_min.reserve(k);
    std::vector<int> k_index;
    k_index.reserve(k);

    k_min.push_back(sorted_values[0].first);
    k_index.push_back(sorted_values[0].second);

    for (int i = 1; i < bound; ++i)
    {
        const value_t &current = sorted_values[i];
        if (not inside_exclusion_zone(k_index, current.second, exclude_zone))
        {
            k_min.push_back(current.first);
            k_index.push_back(current.second);
        }
    }
    return std::make_pair(k_min, k_index);
}