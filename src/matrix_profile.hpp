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
    int exclude = 2; // std::ceil(window_size / 4);
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
    int exclude = 2; //std::ceil(window_size / 4);
    printf("Excluding %d\n", exclude);
    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);

    std::vector<T> row0(n_sequence, T(0));
    std::vector<T> row(n_sequence, T(0));
    std::vector<T> oldrow(n_sequence, T(0));
    std::vector<T> nrow(n_sequence, T(0));

    int max_threads_num = omp_get_max_threads();
    std::vector<T> local_min_values(max_threads_num, std::numeric_limits<T>::max());
    std::vector<int> local_mins_indices(max_threads_num, 0);

    #pragma omp parallel shared(row0, row, nrow, matrix_profile, profile_index, local_min_values, local_mins_indices)
    {
        int threads_num = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int chunk_size = n_sequence / threads_num;
        int start = thread_id * chunk_size;
        int end = (thread_id == threads_num - 1) ? n_sequence : (thread_id + 1) * chunk_size;


        // printf("Parallel block with %d threads and chunks of size %d \n", threads_num, n_sequence / threads_num);

        // First row computed by chunk
        std::span view = std::span(&data[0], window_size);
        for (int j = start; j < end; ++j) {
            const auto distance = dotProduct(view, std::span(&data[j], window_size));
            row0[j] = distance;
            row[j] = distance;

            // local minimum per chunk 
            if (row[j] < local_min_values[thread_id] and (j < - exclude or j > exclude) ) {
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
                nrow[j] = row[j-1] - std::pow(data[j-1]-data[i-1],2) + std::pow(data[j+window_size-1]-data[i+window_size-1],2);
                 //   printf("%d - %d, %f %f %f %f\n", i,j,nrow[j], row[j-1], data[j-1]*data[i-1], data[j+window_size-1]*data[i+window_size-1]);
                // local minimum per chunk
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
                local_min_values[0] = std::min(local_min_values[0], row0[i]);
                for (int k = 0; k < threads_num; ++k) {
                    if (local_min_values[k] < min) {
                        min = local_min_values[k];
                        ind = local_mins_indices[k];
                    }
                    local_min_values[k] = std::numeric_limits<T>::max();
                }
                matrix_profile[i] = std::sqrt(std::abs(min));
                profile_index[i] = ind;
                auto tmp = row;
                row = nrow;
                nrow = oldrow;
                oldrow = tmp;
            }
            #pragma omp barrier
        }
    }
    return std::make_tuple(matrix_profile, profile_index);
}



auto update_row = [&](int i, int j, int global_i, int block_column_start, int window_size, std::vector<T> &row, const std::vector<T> &data) {
    const int global_j = block_column_start + i + j;
    return row[j] - std::pow(data[global_j - 1] - data[global_i - 1], 2) + std::pow(data[global_j + window_size - 1] - data[global_i + window_size - 1], 2);
};

auto update_min = [&](int i, int j, int thread_id, int global_i, int global_j, int exclude) {
    if (row[j] < block_min_value_per_row[thread_id][i] && (global_j < global_i - exclude || global_j > global_i + exclude)) {
        block_min_value_per_row[thread_id][i] = row[j];
        block_min_indice_per_row[thread_id][i] = global_j;
    }
};

template <typename T>
auto computeMatrixProfileSTOMPV2(const std::vector<T> &data,
                               int window_size) {

    const int n_sequence = data.size() - window_size + 1;
    const int exclude = 2;

    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);
    std::vector<T> row0(n_sequence, T(0));


    std::vector<std::vector<T>> block_min_value_per_row;
    std::vector<std::vector<int>> block_min_indice_per_row;   

    printf("Init\n");
    #pragma omp parallel
    {
        const int threads_num = omp_get_num_threads();
        const int thread_id = omp_get_thread_num();
        const int block_width = n_sequence / threads_num;
        const int block_height = threads_num;
        #pragma omp single
        {
            block_min_value_per_row.resize(threads_num);
            block_min_indice_per_row.resize(threads_num);
        }
        block_min_value_per_row[thread_id] = std::vector<T>(block_height, std::numeric_limits<T>::max());
        block_min_indice_per_row[thread_id] = std::vector<int>(block_height, 0);
    }

    #pragma omp parallel shared(matrix_profile, profile_index, block_min_value_per_row, block_min_indice_per_row, row0, data)
    {
        int block_id;
        const int threads_num = omp_get_num_threads();
        const int thread_id = omp_get_thread_num();
        const int block_height = threads_num;

        // Define the width of the block and its starting index
        int block_width;
        int initial_block_start ;
        int leftover = n_sequence % threads_num;
        if (leftover != 0) {
            if (thread_id < leftover) {
                block_width = n_sequence / threads_num + 1;
                initial_block_start = thread_id * block_width;
            } else {
                block_width = n_sequence / threads_num;
                initial_block_start = thread_id * block_width + leftover;
            }
        } else {
            block_width = n_sequence / threads_num;
            initial_block_start = thread_id * block_width;
        }


        const int meta_rows_num = n_sequence%block_height == 0 ? n_sequence / block_height : n_sequence / block_height + 1;

        std::vector<T> row(block_width, T(0));


        // Compute first row
        std::span view = std::span(&data[0], window_size);
        for (int j=0; j < block_width; ++j) {
            const int global_j = initial_block_start + j;
            const auto distance = dotProduct(view, std::span(&data[global_j], window_size));
            row[j] = distance;   
            // Local minimum per block 
            if (row[j] < block_min_value_per_row[thread_id][0] and (global_j < - exclude or global_j > exclude) ) {
                block_min_value_per_row[thread_id][0] = row[j];
                block_min_indice_per_row[thread_id][0] = global_j;
            }
            row0[j] = distance;  // CACHE INVALIDATION
        }
        #pragma omp barrier

        const int last_meta_row_height = (n_sequence%block_height!=0) ? n_sequence%block_height : block_height;
        // For each meta row
        for (int meta_row_id=0; meta_row_id < meta_rows_num; ++meta_row_id) {

            block_id = (meta_row_id + thread_id) % threads_num;
            // Define the block's row start and end: height
            int block_row_start = meta_row_id==0 ? 1 : 0;
            int block_row_end = (meta_row_id==meta_rows_num-1) ? last_meta_row_height : block_height;
            // Define the block's column start and end: width
            int block_column_start = initial_block_start + block_height*meta_row_id;
            int block_column_end = std::min(block_column_start + block_width, n_sequence);
            // Compute all other rows in block
            for (int i=block_row_start; i < block_row_end; ++i) {
                const int global_i = meta_row_id*block_height + i; // Global row index in the distance matrix
                // Semi blocks == last block
                if (block_column_end == n_sequence) {
                    // left semi block
                    if (i > 0) {
                        row[0] = row0[global_i];
                        block_min_value_per_row[thread_id][i] = row[0];
                        for (int j = 1; j < i; ++j) {
                            const int global_j = block_column_start + i + j;
                            row[j] = row[j] - std::pow(data[global_j-1]-data[global_i-1],2) + std::pow(data[global_j+window_size-1]-data[global_i+window_size-1],2);
                            if (row[j] < block_min_value_per_row[thread_id][i] and (global_j < global_i - exclude or global_j > global_i + exclude)) {
                                block_min_value_per_row[thread_id][i] = row[j];
                                block_min_indice_per_row[thread_id][i] = global_j;
                            }
                        }
                    }
                    // right semi block
                    if (i < block_height - 1) {
                        for (int j = 0; j < block_width; ++j) {
                            const int global_j = block_column_start + i + j;
                            row[j] = row[j-1] - std::pow(data[global_j-1]-data[global_i-1],2) + std::pow(data[global_j+window_size-1]-data[global_i+window_size-1],2);
                            // local minimum per row in block
                            if (row[j] < block_min_value_per_row[thread_id][i] and (global_j < global_i - exclude or global_j > global_i + exclude)) {
                                block_min_value_per_row[thread_id][i] = row[j];
                                block_min_indice_per_row[thread_id][i] = global_j;
                            }
                        }
                    }
                }
                else {
                    // Default block
                    for (int j = 0; j < block_width; ++j) {
                        const int global_j = block_column_start + i + j;
                        row[j] = row[j] - std::pow(data[global_j-1]-data[global_i-1],2) + std::pow(data[global_j+window_size-1]-data[global_i+window_size-1],2);
                        // local minimum per row in block
                        if (row[j] < block_min_value_per_row[thread_id][i] and (global_j < global_i - exclude or global_j > global_i + exclude)) {
                            block_min_value_per_row[thread_id][i] = row[j];
                            block_min_indice_per_row[thread_id][i] = global_j;
                        }
                    }
                }
            }
            #pragma omp barrier
            // Compute global minimum per row
            if (thread_id < block_row_end) {
                auto min = matrix_profile[meta_row_id*block_height + thread_id];
                int ind = 0;
                for (int k = 0; k < threads_num; ++k) {
                    if (block_min_value_per_row[k][thread_id] < min) {
                        min = block_min_value_per_row[k][thread_id];
                        ind = block_min_indice_per_row[k][thread_id];
                    }
                    block_min_value_per_row[k][thread_id] = std::numeric_limits<T>::max();
                }
                matrix_profile[meta_row_id*block_height + thread_id] = std::sqrt(T(0)); // CACHE INVALIDATION
                profile_index[meta_row_id*block_height + thread_id] = ind;              // CACHE INVALIDATION
                auto tmp = row;
            }
            #pragma omp barrier
        }
    }
    return std::make_tuple(matrix_profile, profile_index);
}


#endif // MATRIX_PROFILE_H