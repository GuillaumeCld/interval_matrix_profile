#ifndef MATRIX_PROFILE_H
#define MATRIX_PROFILE_H

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath> 
#include <span> 
#include <tuple>
#include <omp.h>    
#include "distance.hpp"

/**
 * @brief Computes the matrix profile of a given time series data using the brute force procedure.
 * 
 * @tparam T The data type of the time series.
 * @param data The input time series data.
 * @param window_size The size of the window for computing the matrix profile.
 * @return std::tuple<std::vector<T>,std::vector<int>> tuple with the matrix profile and its index.
 */
template <typename T>
auto computeMatrixProfileBruteForce(const std::vector<T> &data,
                                    int window_size)
{
    int n_sequence = data.size() - window_size + 1;
    int exclude = std::round(window_size / 2);
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);
    
    #pragma omp parallel for shared(matrix_profile, profile_index)
    for (int i = 0; i < n_sequence; ++i) {
        std::span view = std::span(&data[i], window_size);
        for (int j = 0; j < n_sequence; ++j) {
            // Compute Euclidean distance for the current sequence
            const auto distance = euclideanDistance(view, std::span(&data[j], window_size));
            if (distance < matrix_profile[i] and (j < i - exclude or j > i + exclude) ) {
                matrix_profile[i] = distance;
                profile_index[i] = j; // Update index
            }
        }
    }

    return std::make_tuple(matrix_profile, profile_index);
}

/**
 * @brief 
 * 
 * @tparam T 
 * @param data 
 * @param window_size 
 * @return std::tuple<std::vector<T>,std::vector<int>> tuple with the matrix profile and its index.
 */
template <typename T>
auto computeMatrixProfileSTOMP(const std::vector<T> &data,
                               int window_size) {
    int n_sequence = data.size() - window_size + 1;
    int exclude = std::round(window_size / 2);
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

    std::vector<T> row0(n_sequence, T(0));
    std::vector<T> row(n_sequence, T(0));
    std::vector<T> nrow(n_sequence, T(0));

    int max_threads_num = omp_get_max_threads();
    std::vector<T> local_min_values(max_threads_num, std::numeric_limits<T>::max());
    std::vector<int> local_mins_indices(max_threads_num, 0);

    #pragma omp parallel shared(row0, row, nrow, matrix_profile, profile_index, local_min_values, local_mins_indices)
    {
        int threads_num = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int chunk_size = n_sequence / threads_num;
        int start = (thread_id == 0) ? exclude+1 : thread_id * chunk_size;
        int end = (thread_id == threads_num - 1) ? n_sequence : (thread_id + 1) * chunk_size;



        // printf("Parallel block with %d threads and chunks of size %d \n", threads_num, n_sequence / threads_num);

        // First row computed by chunk
        std::span view = std::span(&data[0], window_size);
        for (int j = start; j < end; ++j) {
            const auto distance = dotProduct(view, std::span(&data[j], window_size));
            row0[j] = distance;
            row[j] = distance;

            // local minimum per chunk 
            if (row[j] < local_min_values[thread_id] ) {
                local_min_values[thread_id] = row[j];
                local_mins_indices[thread_id] = j;
            }
            
        }
        
        #pragma omp barrier
        #pragma omp single
        { 
            // global minimum
            auto min = matrix_profile[0];
            int ind = 0;
            for (int k = 0; k < threads_num; ++k) {
                if (local_min_values[k] < min) {
                    min = local_min_values[k];
                    ind = local_mins_indices[k];
                }
                local_min_values[k] = std::numeric_limits<T>::max();
            }
            matrix_profile[0] = std::sqrt(min);
            profile_index[0] = ind;

        }
        #pragma omp barrier
        // Compute all other rows
        for (int i = 1; i < n_sequence; ++i) {
            // Compute the i-th row of the distance matrix based on the (i-1)-th and the first one (row0)
            // First chunk
            if (thread_id == 0) {
                nrow[0] = row0[i];
                start = 1;
            }
            // Diagonal recurrence for all chunks
            for (int j = start; j < end; ++j) {
                nrow[j] = row[j-1] - data[j-1]*data[i-1] + data[j+window_size-1]*data[i+window_size-1];
                // printf("%d - %d, %f %f %f %f\n", i,j,nrow[j], row[j-1], data[j-1]*data[i-1], data[j+window_size-1]*data[i+window_size-1]);

                if (nrow[j] < local_min_values[thread_id] and (j < i - exclude or j > i + exclude)) {
                    local_min_values[thread_id] = nrow[j];
                    local_mins_indices[thread_id] = j;
                }
            }
            #pragma omp barrier
            #pragma omp single
            { 
                // Global minimum on row
                auto min = matrix_profile[i];
                int ind = 0;
                for (int k = 0; k < threads_num; ++k) {
                    if (local_min_values[k] < min) {
                        min = local_min_values[k];
                        ind = local_mins_indices[k];
                    }
                    local_min_values[k] = std::numeric_limits<T>::max();
                }
                matrix_profile[i] = std::sqrt(min);
                // printf("MP %f \n", matrix_profile[i]);
                profile_index[i] = ind;
                row = nrow;
            }
            #pragma omp barrier
        }
    }
    return std::make_tuple(matrix_profile, profile_index);
}

#endif // MATRIX_PROFILE_H