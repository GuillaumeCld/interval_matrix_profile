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
auto inteval_matrix_profile_brute_force(std::vector<T> &time_series,
                                        const int window_size,
                                        std::vector<int> const &period_starts,
                                        const int interval_length,
                                        const int exclude)
{
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);
    const int half_interval = interval_length / 2;
    bool is_periodic = window_size <= half_interval;
    const int n_periods_i = period_starts.size();

    // #pragma omp parallel shared(time_series, matrix_profile, profile_index)
    for (int i_period = 0; i_period < n_periods_i; ++i_period)
    {
        const int i_start = period_starts[i_period];
        int tmp_end;
        if (i_period == n_periods_i - 1)
        {
            tmp_end = n_sequence;
        }
        else
        {
            tmp_end = period_starts[i_period + 1];
        }
        const int i_end = std::min(tmp_end, n_sequence);
        // #pragma omp for
        for (int i = i_start; i < i_end; ++i)
        {
            const int place_in_period = std::min(i - i_start, 365);
            auto view = std::span(&time_series[i], window_size);
            auto min = std::numeric_limits<T>::max();
            int min_index = 0;

            for (int j_period = 0; j_period < n_periods_i; ++j_period)
            {
                if (j_period == n_periods_i - 1 and period_starts[j_period] + place_in_period > n_sequence)
                {
                    break;
                }
                const int j_start = std::max(period_starts[j_period] + place_in_period - half_interval, 0);
                const int j_end = std::min(period_starts[j_period] + place_in_period + half_interval, n_sequence);
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
            if (is_periodic)
            {
                const int j_start = n + 1 + i - half_interval;
                const int j_end = std::min(n + 1 + i + half_interval, n_sequence);
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

    return std::make_tuple(matrix_profile, profile_index);
}

template <typename T>
auto inteval_matrix_profile_STOMP(std::vector<T> &time_series,
                                  const int window_size,
                                  std::vector<int> const &period_starts,
                                  const int interval_length,
                                  const int exclude)
{
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int metarows = period_starts.size();
    const int half_interval = interval_length / 2;

    const int nb_blocks = (window_size > half_interval) ? metarows : metarows + 1;
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

    std::vector<std::vector<min_pair<T>>> block_min_pair_per_row(nb_blocks);
    std::vector<T> first_row(n_sequence);
    std::vector<block<T>> current_blocks(nb_blocks);
    std::vector<std::vector<T>> previous_blocks(nb_blocks);
    int block_height;
    int block_width;
    int block_i = 0;
    int block_j;
    int period_width;

    std::span view = std::span(&time_series[0], window_size);
    for (int j = 0; j < n_sequence; ++j)
    {
        first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
    }
    for (int metarow = 0; metarow < metarows; ++metarow)
    {
        // Compute the height of the current block (last block may be shorter)
        if (metarow == metarows - 1)
        {
            block_height = n_sequence - period_starts[metarow];
        }
        else
        {
            block_height = period_starts[metarow + 1] - period_starts[metarow];
        }
        for (int column = 0; column < nb_blocks; ++column)
        {
            if (column < metarows)
            {
                if (column == metarows - 1)
                {
                    block_width = std::min(interval_length, n_sequence - (period_starts[column] - half_interval));
                    period_width = n + 1 - period_starts[column];
                }
                else
                {
                    block_width = interval_length;
                    period_width = period_starts[column + 1] - period_starts[column];
                }
                block_j = period_starts[column] - half_interval;
            }
            else
            {
                block_j = n + 1 - half_interval;
                period_width = block_height;
                if (block_j >= n_sequence)
                {
                    break;
                }
            }
            printf("Height width %d %d \n", block_height, period_width);
            std::span<T> initial_row;
            std::vector<T> tmp(block_width, 0);

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
                std::span<T> view = std::span(&time_series[block_i - 1], window_size);

                if (block_j < 0)
                {
                    for (int j = half_interval; j < block_width; ++j)
                    {
                        tmp[j] = dotProduct(view, std::span(&time_series[block_j + j - 1], window_size));
                    }
                    initial_row = std::span(tmp.data(), block_width);
                }
                else
                {
                    const int previous_block = std::max(column - 1, 0);
                    initial_row = std::span(previous_blocks[previous_block].data(), block_width);
                }
            }
            block<T> block(n_sequence,
                           window_size,
                           exclude,
                           block_i,
                           block_j,
                           column,
                           block_width,
                           block_height,
                           first_row,
                           initial_row,
                           time_series);
            current_blocks[column] = std::move(block);
            current_blocks[column].STOMP();
            // retrieve the minimums per row
            block_min_pair_per_row[column] = current_blocks[column].get_local_min_rows();
        }

        // Compute the global minimums per row
        for (int i = 0; i < block_height; ++i)
        {
            T min{std::numeric_limits<T>::max()};
            int ind{-1};
            for (int k = 0; k < nb_blocks; ++k)
            {
                const min_pair<T> min_pair = block_min_pair_per_row[k][i];
                if (min_pair.value < min)
                {
                    min = min_pair.value;
                    ind = min_pair.index;
                }
            }

            matrix_profile[block_i + i] = std::sqrt(std::abs(min));
            profile_index[block_i + i] = ind;
        }
        for (int column = 0; column < nb_blocks; ++column)
        {
            if (column < metarows)
            {
                if (column == metarows - 1)
                {
                    block_width = std::min(interval_length, n_sequence - (period_starts[column] - half_interval));
                    period_width = n + 1 - period_starts[column];
                }
                else
                {
                    block_width = interval_length;
                    period_width = period_starts[column + 1] - period_starts[column];
                }
                block_j = period_starts[column] - half_interval;
            }
            else
            {
                block_j = n + 1 - half_interval;
                period_width = block_height;
                if (block_j >= n_sequence)
                {
                    break;
                }
            }
            if (block_height != period_width)
            {
                if (block_height < period_width)
                {
                    const int i = block_i + block_height - 1;
                    const int first_j = block_j + block_height - 1;
                    const int last_j = first_j + block_width - 1;

                    auto tmp_row = current_blocks[column].get_row();
                    for (int j = 1; j < block_width; ++j)
                    {
                        previous_blocks[column][j - 1] = tmp_row[j];
                    }
                    previous_blocks[column][block_width - 1] = dotProduct(std::span(&time_series[i], block_width), std::span(&time_series[last_j], block_width));
                }
                else if (block_height > period_width)
                {
                    printf(" (%d, %d) \n", metarow, column);
                    const int i = block_i + block_height - 1;
                    const int first_j = block_j + block_height - 1;
                    const int last_j = first_j + block_width;
                    auto const &tmp_row = current_blocks[column].get_row();

                    for (int j = 0; j < block_width - 1; ++j)
                    {
                        previous_blocks[column][j + 1] = tmp_row[j];
                    }
                    previous_blocks[column][0] = dotProduct(std::span(&time_series[i], block_width), std::span(&time_series[first_j - 1], block_width));
                }
            }
            else
            {
                previous_blocks[column] = std::move(current_blocks[column].get_row());
            }
        }
        block_i += block_height;
    }
    return std::make_tuple(matrix_profile, profile_index);
}

template <typename T>
auto inteval_matrix_profile_STOMP_bf(std::vector<T> &time_series,
                                     const int window_size,
                                     std::vector<int> const &period_starts,
                                     const int interval_length,
                                     const int exclude)
{
    const int n = time_series.size();
    const int n_sequence = n - window_size + 1;
    const int metarows = period_starts.size();
    const int half_interval = interval_length / 2;

    const int nb_blocks = (window_size > half_interval) ? metarows : metarows + 1;
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

    std::vector<std::vector<min_pair<T>>> block_min_pair_per_row(nb_blocks);
    std::vector<T> first_row(n_sequence);
    std::vector<block<T>> current_blocks(nb_blocks);
    std::vector<std::vector<T>> previous_blocks(nb_blocks);
    int block_height;
    int block_width;
    int block_i = 0;
    int block_j;
    int period_width;

    std::span view = std::span(&time_series[0], window_size);
    for (int j = 0; j < n_sequence; ++j)
    {
        first_row[j] = dotProduct(view, std::span(&time_series[j], window_size));
    }
    for (int metarow = 0; metarow < metarows; ++metarow)
    {
        // Compute the height of the current block (last block may be shorter)
        if (metarow == metarows - 1)
        {
            block_height = n_sequence - period_starts[metarow];
        }
        else
        {
            block_height = period_starts[metarow + 1] - period_starts[metarow];
        }
        for (int column = 0; column < nb_blocks; ++column)
        {
            if (column < metarows)
            {
                if (column == metarows - 1)
                {
                    block_width = std::min(interval_length, n_sequence - (period_starts[column] - half_interval));
                    period_width = n + 1 - period_starts[column];
                }
                else
                {
                    block_width = interval_length;
                    period_width = period_starts[column + 1] - period_starts[column];
                }
                block_j = period_starts[column] - half_interval;
            }
            else
            {
                block_j = n + 1 - half_interval;
                period_width = block_height;
                if (block_j >= n_sequence)
                {
                    break;
                }
            }
            std::span<T> initial_row;
            std::vector<T> tmp(block_width, 0);
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
                std::span<T> view = std::span(&time_series[block_i - 1], window_size);

                if (block_j < 0)
                {
                    for (int j = half_interval; j < block_width; ++j)
                    {
                        tmp[j] = dotProduct(view, std::span(&time_series[block_j + j - 1], window_size));
                    }
                    initial_row = std::span(tmp.data(), block_width);
                }
                else
                {
                    for (int j = 0; j < block_width; ++j)
                    {
                        tmp[j] = dotProduct(view, std::span(&time_series[block_j + j - 1], window_size));
                    }
                    initial_row = std::span(tmp.data(), block_width);
                }
            }
            block<T> block(n_sequence,
                           window_size,
                           exclude,
                           block_i,
                           block_j,
                           column,
                           block_width,
                           block_height,
                           first_row,
                           initial_row,
                           time_series);

            current_blocks[column] = std::move(block);
            current_blocks[column].STOMP();
            // retrieve the minimums per row
            block_min_pair_per_row[column] = current_blocks[column].get_local_min_rows();
        }
        // Compute the global minimums per row
        for (int i = 0; i < block_height; ++i)
        {
            T min{std::numeric_limits<T>::max()};
            int ind{-1};
            for (int k = 0; k < nb_blocks; ++k)
            {
                const min_pair<T> min_pair = block_min_pair_per_row[k][i];
                if (min_pair.value < min)
                {
                    min = min_pair.value;
                    ind = min_pair.index;
                }
            }

            matrix_profile[block_i + i] = std::sqrt(std::abs(min));
            profile_index[block_i + i] = ind;
        }
        block_i += block_height;
    }
    return std::make_tuple(matrix_profile, profile_index);
}
