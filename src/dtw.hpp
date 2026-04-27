#pragma once

#include <iostream>
#include <span>

/**
 * @brief Compute the Dynamic Time Warping (DTW) distance between two vectors x, y.
 *
 * @tparam T type of the elements in the vectors
 * @tparam N size of the vectors
 * @param x the first vector
 * @param y the second vector
 * @return T DTW distance
 */
template <typename T, std::size_t N>
inline auto dtw(std::span<T, N> const &x, std::span<T, N> const &y) -> T
{
    static_assert(std::is_floating_point<T>::value, " works only with floating values");

    auto const n{x.size()};
    auto const m{y.size()};

    std::vector<T> dtw_matrix((n + 1) * (m + 1), std::numeric_limits<T>::max());
    dtw_matrix[0] = T(0);

    for (int i = 1; i <= n; ++i)
    {
        for (int j = 1; j <= m; ++j)
        {
            T const cost{x[i - 1] - y[j - 1]};

            dtw_matrix[i * (m + 1) + j] = cost * cost + std::min({dtw_matrix[(i - 1) * (m + 1) + j], dtw_matrix[i * (m + 1) + j - 1], dtw_matrix[(i - 1) * (m + 1) + j - 1]});
        }
    }
    return dtw_matrix[n * (m + 1) + m];
}
template <typename array_value_t, typename array_index_t>
auto imp_bf_ab_dtw(array_value_t &time_series_A,
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
                auto view_B = std::span(&time_series_B[j], window_size);

                const value_t dist = dtw(view_A, view_B);

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
                const int j_end   = std::min(base + place_in_period + half_interval + 1,
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

            matrix_profile[i] = std::sqrt(min_val); // optional: keep consistent with your previous outputs
            profile_index[i] = min_index;
        }
    }

    return std::make_pair(matrix_profile, profile_index);
}