#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <omp.h>
#include "distance.hpp"
#include "square_block.hpp"

/**
 * @brief Computes the seasonal matrix profile using a brute force approach.
 *
 * This function calculates the seasonal matrix profile for a given time series data using a brute force approach.
 *
 * @tparam T The data type of the time series data.
 * @param data The time series data.
 * @param window_size The size of the sliding window used to calculate the matrix profile.
 * @param exclude The exclusion zone around each subsequence.
 * @param seasons The seasons for which the matrix profile is calculated.
 * @return A tuple containing the matrix profile and the corresponding profile indices.
 */
template <typename T>
auto seasonal_matrix_profile_brute_force(std::vector<T> &data,
                                         const int window_size,
                                         const int exclude,
                                         std::vector<std::vector<std::pair<int, int>>> const &seasons)
{
    const int n_sequence = data.size() - window_size + 1;
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

#pragma omp parallel shared(data, matrix_profile, profile_index)
    for (auto const &season : seasons)
    {
        for (auto const &pair_row : season)
        {
            const int start_row{pair_row.first};
            const int end_row{pair_row.second};
#pragma omp for
            for (int i = start_row; i < end_row; ++i)
            {
                T min{std::numeric_limits<T>::max()};
                int min_index{-1};

                std::span<T> view = std::span(&data[i], window_size);
                for (const auto &pair_column : season)
                {
                    const int start_column{pair_column.first};
                    const int end_column{pair_column.second};
                    for (int j = start_column; j < end_column; ++j)
                    {
                        // Compute Euclidean distance for the current sequence
                        const T distance = dotProduct(view, std::span(&data[j], window_size));
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
    }
    return std::make_pair(matrix_profile, profile_index);
}

/**
 * @brief Calculates the seasonal matrix profile using a brute-force blocking approach.
 *
 * This function calculates the seasonal matrix profile for a given time series data using a brute-force blocking approach.
 *
 * @tparam T The data type of the time series data.
 * @param data The time series data.
 * @param window_size The size of the sliding window used to calculate the matrix profile.
 * @param exclude The number of nearest neighbors to exclude when calculating the matrix profile.
 * @param seasons The seasons of the time series data, represented as a vector of vector of pairs. Each pair represents the start and end indices of a season.
 * @return A tuple containing the matrix profile and the corresponding profile indices.
 */
template <typename T>
auto seasonal_matrix_profile_brute_force_blocking(std::vector<T> &data,
                                                  const int window_size,
                                                  const int exclude,
                                                  std::vector<std::vector<std::pair<int, int>>> const &seasons)
{
    const int n_sequence = data.size() - window_size + 1;
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);
#pragma omp parallel default(none) \
    shared(data, matrix_profile, profile_index, n_sequence, window_size, exclude, seasons)
#pragma omp single
    {

        for (auto const &season : seasons)
        {
            for (auto const &pair_row : season)
            {
#pragma omp task default(none)                                                    \
    shared(data, n_sequence, window_size, exclude, matrix_profile, profile_index) \
    firstprivate(season, pair_row)                                                \
    untied
                {
                    const int start_row{pair_row.first};
                    const int end_row{pair_row.second};
                    const int height{end_row - start_row};

                    std::vector<min_pair<T>> local_min_row(height, min_pair{-1, std::numeric_limits<T>::max()});

                    for (const auto &pair_column : season)
                    {
                        const int start_column{pair_column.first};
                        const int end_column{pair_column.second};
                        const int width{end_column - start_column};

                        square_block<T> square_block(
                            n_sequence,
                            window_size,
                            exclude,
                            start_row,
                            start_column,
                            width,
                            height,
                            local_min_row,
                            data);

                        square_block.brute_force();
                        local_min_row = std::move(square_block.get_local_min_rows());
                    }
                    for (int i = 0; i < height; ++i)
                    {
                        matrix_profile[start_row + i] = std::sqrt(local_min_row[i].value);
                        profile_index[start_row + i] = local_min_row[i].index;
                    }
                }
            }
        }
    }
    return std::make_pair(matrix_profile, profile_index);
}

/**
 * @brief Calculates the seasonal matrix profile using the STOMP algorithm with blocking.
 *
 * This function calculates the seasonal matrix profile for a given data sequence using the STOMP algorithm with blocking.
 *
 * @tparam T The data type of the elements in the data sequence.
 * @param data The input data sequence.
 * @param window_size The size of the sliding window used for calculating the matrix profile.
 * @param exclude The number of nearest neighbors to exclude when calculating the matrix profile.
 * @param seasons The seasonal patterns represented as a vector of vector of pairs, where each pair represents the start and end indices of a season.
 * @return A tuple containing the matrix profile and the profile index.
 */
template <typename T>
auto seasonal_matrix_profile_STOMP_blocking(std::vector<T> &data,
                                            const int window_size,
                                            const int exclude,
                                            std::vector<std::vector<std::pair<int, int>>> const &seasons)
{
    const int n_sequence = data.size() - window_size + 1;
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

    #pragma omp parallel default(none) \
    shared(data, matrix_profile, profile_index, n_sequence, window_size, exclude, seasons)
    #pragma omp single
    {
        for (const auto &season : seasons)
        {

            for (const auto &pair_row : season)
            {

                #pragma omp task default(none)                                                \
                shared(data, n_sequence, window_size, exclude, matrix_profile, profile_index) \
                firstprivate(season, pair_row)                                                \
                untied
                {
                    const int start_row{pair_row.first};
                    const int end_row{pair_row.second};
                    const int height{end_row - start_row};

                    std::vector<min_pair<T>> local_min_row(height, min_pair<T>{-1, std::numeric_limits<T>::max()});
                    for (const auto &pair_column : season)
                    {
                        const int start_column{pair_column.first};
                        const int end_column{pair_column.second};
                        const int width{end_column - start_column};

                        square_block<T> square_block(
                            n_sequence,
                            window_size,
                            exclude,
                            start_row,
                            start_column,
                            width,
                            height,
                            local_min_row,
                            data);

                        square_block.STOMP();
                        local_min_row = std::move(square_block.get_local_min_rows());
                    }
                    for (int i = 0; i < height; ++i)
                    {
                        matrix_profile[start_row + i] = std::sqrt(std::abs(local_min_row[i].value));
                        profile_index[start_row + i] = local_min_row[i].index;
                    }
                }
            }
        }
    }
    return std::make_pair(matrix_profile, profile_index);
}