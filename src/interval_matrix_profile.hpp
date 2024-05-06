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

template <typename T>
auto inteval_matrix_profile_brute_force(const std::vector<T> &data,
                                        const int window_size,
                                        std::vector<int> period_starts,
                                        const int interval_length)
{
    int n_sequence = data.size() - window_size + 1;
    int exclude = 2; // std::ceil(window_size / 4);
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

    #pragma omp parallel shared(data, matrix_profile, profile_index)
    for (int i_period = 0; i_period < period_starts.size(); ++i_period)
    {
        int i_start = period_starts[i_period];
        int i_end = std::min(period_starts[i_period + 1], n_sequence);

        #pragma omp for
        for (int i = i_start; i < i_end; ++i)
        {
            int place_in_period = i - i_start;
            auto view = std::span(&data[i], window_size);
            auto min = std::numeric_limits<T>::max();
            int min_index = 0;


            for (int j_period = 0; j_period < period_starts.size(); ++j_period)
            {
                if (j_period == period_starts.size() - 1 and period_starts[j_period] + place_in_period > n_sequence)
                {
                    break;
                }
                int j_start = std::max(period_starts[j_period] + place_in_period - interval_length/2, 0);
                int j_end = std::min(period_starts[j_period] + place_in_period + interval_length/2, n_sequence);

                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const auto distance = dotProduct(view, std::span(&data[j], window_size));
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

    return std::make_tuple(matrix_profile, profile_index);
}
