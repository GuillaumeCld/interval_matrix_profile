#pragma once
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <omp.h>
#include "distance.hpp"
#include "gen_block.hpp"
/**
 * @brief Compute the Interval Matrix Profile (AB-join) and index with BIMP blocks.
 *
 * @tparam query_array_t
 * @tparam target_array_t
 * @tparam array_index_t
 * @param query_series row time series (A)
 * @param target_series column time series (B)
 * @param window_size subsequence length
 * @param query_period_starts start index of each query period (rows)
 * @param target_period_starts start index of each target period (columns)
 * @param interval_length interval length (L)
 * @return pair of matrix profile and profile index (for query subsequences)
 */
template <typename array_value_t, typename array_index_t>
auto outer_BIMP(array_value_t& query_series,
                array_value_t& target_series,
                const int window_size,
                array_index_t& query_period_starts,
                array_index_t& target_period_starts,
                const int interval_length)
{
    using value_t = typename array_value_t::value_type;
    using index_t = typename array_index_t::value_type;
    using pair_t  = min_pair<value_t>;

    const int n_query = static_cast<int>(query_series.size());
    const int n_target = static_cast<int>(target_series.size());
    const int n_query_sequence = n_query - window_size + 1;
    const int n_target_sequence = n_target - window_size + 1;

    const int metarows = static_cast<int>(query_period_starts.size());
    const int metacols = static_cast<int>(target_period_starts.size());
    const int half_interval = interval_length / 2;
    const int nb_blocks = (window_size > half_interval) ? metacols : metacols + 1;

    std::vector<value_t> matrix_profile(n_query_sequence, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_query_sequence, static_cast<index_t>(-1));
    std::vector<value_t> query_first_row(n_query_sequence);
    std::vector<value_t> target_first_row(n_target_sequence);

    int block_height;
    int block_width;
    int block_i;
    int block_j;

    #pragma omp parallel default(none) \
    shared(query_series, target_series, matrix_profile, profile_index, query_first_row, target_first_row, query_period_starts, target_period_starts, std::cout) \
    firstprivate(n_query, n_target, n_query_sequence, n_target_sequence, metarows, metacols, interval_length, half_interval, nb_blocks, window_size) \
    private(block_height, block_width, block_i, block_j)
    {
        

        #pragma omp single
        {
            // Query first row: dot(query[0:m], target[j:j+m])
            std::span<const value_t> query0(&query_series[0], window_size);
            // #pragma omp taskloop
            for (int j = 0; j < n_target_sequence; ++j)
            {
                target_first_row[j] = dotProduct(query0, std::span<const value_t>(&target_series[j], window_size));
            }
            // Target first row: dot(target[0:m], query[i:i+m])
            std::span<const value_t> target0(&target_series[0], window_size);
            // #pragma omp taskloop
            for (int i = 0; i < n_query_sequence; ++i)
            {
                query_first_row[i] = dotProduct(target0, std::span<const value_t>(&query_series[i], window_size));
            }
        
        

            #pragma omp taskwait

            // #pragma omp taskloop untied
            for (int metarow = 0; metarow < metarows; ++metarow)
            {
                
                
                block_i = query_period_starts[metarow];
                block_height = (metarow == metarows - 1)
                                 ? n_query_sequence - query_period_starts[metarow]
                                 : query_period_starts[metarow + 1] - query_period_starts[metarow];

                std::vector<pair_t> local_min_row(block_height, {static_cast<index_t>(-1), std::numeric_limits<value_t>::max()});

                for (int column = -1; column < nb_blocks; ++column)
                {
                    if (column == -1)
                    {
                        block_width = interval_length + 1;
                        block_j = -block_height - half_interval;
                    }
                    else if (column < metacols)
                    {
                        block_width = (column == metacols - 1)
                                        ? std::min(interval_length + 1, n_target_sequence - (target_period_starts[column] - half_interval))
                                        : interval_length + 1;
                        block_j = target_period_starts[column] - half_interval;
                    }
                    else
                    {
                        block_j = n_target - half_interval + 1;
                        if (block_j > n_target_sequence)
                        {
                            break;
                        }
                    }

                    gblock<value_t> block(n_target_sequence,
                                              window_size,
                                              block_i,
                                              block_j,
                                              column,
                                              block_width,
                                              block_height,
                                              query_first_row,
                                              target_first_row,
                                              query_series,
                                              target_series,
                                              local_min_row);
                    block.compute();
                    local_min_row = std::move(block.get_local_min_rows());
                }

                for (int i = 0; i < block_height; ++i)
                {
                    matrix_profile[block_i + i] = std::sqrt(std::abs(local_min_row[i].value));
                    profile_index[block_i + i] = local_min_row[i].index;

                }
            }
        }
    }

    return std::make_pair(matrix_profile, profile_index);
}


template <typename array_value_t, typename array_index_t>
auto imp_bf_ab(array_value_t &time_series_A,
               array_value_t &time_series_B,
               const int window_size,
               array_index_t const &period_starts_A,
               array_index_t const &period_starts_B,
               const int interval_length)
{
    using value_t = typename array_value_t::value_type;
    using index_t = typename array_index_t::value_type;

    const int n_A = time_series_A.size();
    const int n_B = time_series_B.size();

    const int n_sequence_A = n_A - window_size + 1;
    const int n_sequence_B = n_B - window_size + 1;

    std::vector<value_t> matrix_profile(n_sequence_A, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_sequence_A, 0);

    const int half_interval = interval_length / 2;
    const bool is_periodic = window_size <= half_interval;

    const int n_periods_A = period_starts_A.size();
    const int n_periods_B = period_starts_B.size();

#pragma omp parallel for schedule(static)
    for (int i_period = 0; i_period < n_periods_A; ++i_period)
    {
        const int i_start = period_starts_A[i_period];
        const int i_end = (i_period == n_periods_A - 1)
                              ? n_sequence_A
                              : period_starts_A[i_period + 1];

        for (int i = i_start; i < i_end; ++i)
        {
            const int place_in_period = i - i_start;
            const int i_pos = place_in_period;

            auto view_A = std::span(&time_series_A[i], window_size);

            value_t min_val = std::numeric_limits<value_t>::max();
            index_t min_index = 0;

            // --- Main interval search over B ---
            for (int j_period = 0; j_period < n_periods_B; ++j_period)
            {
                const int base = period_starts_B[j_period];

                const int j_start = std::max(base + place_in_period - half_interval, 0);
                const int j_end   = std::min(base + place_in_period + half_interval + 1,
                                             n_sequence_B);

                for (int j = j_start; j < j_end; ++j)
                {
                    const auto distance = dotProduct(
                        view_A,
                        std::span(&time_series_B[j], window_size)
                    );

                    if (distance < min_val)
                    {
                        min_val = distance;
                        min_index = j;
                    }
                }
            }

            // --- Left edge handling (wrap to beginning of B) ---
            if (i_end - i <= half_interval)
            {
                const int overflow = half_interval - (i_end - i);

                const int j_start = 0;
                const int j_end = std::min(overflow + 1, n_sequence_B);

                for (int j = j_start; j < j_end; ++j)
                {
                    const auto distance = dotProduct(
                        view_A,
                        std::span(&time_series_B[j], window_size)
                    );

                    if (distance < min_val)
                    {
                        min_val = distance;
                        min_index = j;
                    }
                }
            }

            // --- Right edge handling (wrap to end of B) ---
            if (is_periodic && i_pos <= half_interval)
            {
                int j_start = n_sequence_B + i_pos - half_interval;
                j_start = std::max(j_start, 0);

                const int j_end = n_sequence_B;

                for (int j = j_start; j < j_end; ++j)
                {
                    const auto distance = dotProduct(
                        view_A,
                        std::span(&time_series_B[j], window_size)
                    );

                    if (distance < min_val)
                    {
                        min_val = distance;
                        min_index = j;
                    }
                }
            }

            matrix_profile[i] = std::sqrt(min_val);
            profile_index[i] = min_index;
        }
    }

    return std::make_pair(matrix_profile, profile_index);
}

template <typename array_value_t, typename array_index_t>
auto imp_bf_ab_znorm(array_value_t &time_series_A,
                     array_value_t &time_series_B,
                     const int window_size,
                     array_index_t const &period_starts_A,
                     array_index_t const &period_starts_B,
                     const int interval_length)
{
    using value_t = typename array_value_t::value_type;
    using index_t = typename array_index_t::value_type;

    const int n_A = time_series_A.size();
    const int n_B = time_series_B.size();

    const int n_seq_A = n_A - window_size + 1;
    const int n_seq_B = n_B - window_size + 1;

    std::vector<value_t> matrix_profile(n_seq_A, std::numeric_limits<value_t>::max());
    std::vector<index_t> profile_index(n_seq_A, 0);

    const int half_interval = interval_length / 2;
    const bool is_periodic = window_size <= half_interval;

    const int n_periods_A = period_starts_A.size();
    const int n_periods_B = period_starts_B.size();

    // --- Precompute mean and std using prefix sums ---
    auto compute_stats = [&](const array_value_t &ts,
                             std::vector<value_t> &mean,
                             std::vector<value_t> &std)
    {
        const int n = ts.size();
        std::vector<value_t> prefix(n + 1, 0), prefix_sq(n + 1, 0);

        for (int i = 0; i < n; ++i)
        {
            prefix[i + 1] = prefix[i] + ts[i];
            prefix_sq[i + 1] = prefix_sq[i] + ts[i] * ts[i];
        }

        for (int i = 0; i < (int)mean.size(); ++i)
        {
            value_t sum = prefix[i + window_size] - prefix[i];
            value_t sum_sq = prefix_sq[i + window_size] - prefix_sq[i];

            mean[i] = sum / window_size;
            value_t var = (sum_sq / window_size) - mean[i] * mean[i];
            std[i] = (var > 0) ? std::sqrt(var) : 0;
        }
    };

    std::vector<value_t> mean_A(n_seq_A), std_A(n_seq_A);
    std::vector<value_t> mean_B(n_seq_B), std_B(n_seq_B);

    compute_stats(time_series_A, mean_A, std_A);
    compute_stats(time_series_B, mean_B, std_B);

#pragma omp parallel for schedule(static)
    for (int i_period = 0; i_period < n_periods_A; ++i_period)
    {
        const int i_start = period_starts_A[i_period];
        const int i_end = (i_period == n_periods_A - 1)
                              ? n_seq_A
                              : period_starts_A[i_period + 1];

        for (int i = i_start; i < i_end; ++i)
        {
            const int place_in_period = i - i_start;
            const int i_pos = place_in_period;

            auto view_A = std::span(&time_series_A[i], window_size);

            value_t min_val = std::numeric_limits<value_t>::max();
            index_t min_index = 0;

            auto eval = [&](int j)
            {
                if (std_A[i] == 0 || std_B[j] == 0)
                    return; // skip constant subsequences

                const auto dot = dotProduct(view_A,
                                            std::span(&time_series_B[j], window_size));

                const value_t denom = window_size * std_A[i] * std_B[j];
                value_t corr = (dot - window_size * mean_A[i] * mean_B[j]) / denom;

                // numerical safety
                corr = std::min<value_t>(1, std::max<value_t>(-1, corr));

                value_t dist = 2 * window_size * (1 - corr);

                if (dist < min_val)
                {
                    min_val = dist;
                    min_index = j;
                }
            };

            // --- Main interval ---
            for (int j_period = 0; j_period < n_periods_B; ++j_period)
            {
                const int base = period_starts_B[j_period];

                const int j_start = std::max(base + place_in_period - half_interval, 0);
                const int j_end = std::min(base + place_in_period + half_interval + 1,
                                           n_seq_B);

                for (int j = j_start; j < j_end; ++j)
                    eval(j);
            }

            // --- Left edge wrap ---
            if (i_end - i <= half_interval)
            {
                const int overflow = half_interval - (i_end - i);
                const int j_end = std::min(overflow + 1, n_seq_B);

                for (int j = 0; j < j_end; ++j)
                    eval(j);
            }

            // --- Right edge wrap ---
            if (is_periodic && i_pos <= half_interval)
            {
                int j_start = n_seq_B + i_pos - half_interval;
                j_start = std::max(j_start, 0);

                for (int j = j_start; j < n_seq_B; ++j)
                    eval(j);
            }

            matrix_profile[i] = std::sqrt(min_val);
            profile_index[i] = min_index;
        }
    }

    return std::make_pair(matrix_profile, profile_index);
}