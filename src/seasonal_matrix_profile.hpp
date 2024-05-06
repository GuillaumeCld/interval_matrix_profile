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
auto seasonal_matrix_profile_brute_force(const std::vector<T> &data,
                                        int window_size, 
                                        std::vector<std::vector<std::pair<int, int>>> seasons)
{
    int n_sequence = data.size() - window_size + 1;
    int exclude = 2; // std::ceil(window_size / 4);
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

    #pragma omp parallel shared(data, matrix_profile, profile_index)
    for (const auto &season : seasons)
    {
        for (const auto &pair : season)
        {
            int start = pair.first;
            int end = pair.second;
            #pragma omp for
            for (int i = start; i < end; ++i)
            {
                auto view = std::span(&data[i], window_size);
                auto min = std::numeric_limits<T>::max();
                int min_index = 0;
                for (int j = start; j < end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const auto distance = dotProduct(view, std::span(&data[j], window_size));
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
                auto block_min = std::sqrt(min);
                if (block_min < matrix_profile[i])
                {
                    matrix_profile[i] = block_min;
                    profile_index[i] = min_index;
                }
            }
        }
    }
    return std::make_tuple(matrix_profile, profile_index);
}
