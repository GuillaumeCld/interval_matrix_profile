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

    #pragma omp parallel for shared(matrix_profile, profile_index, data)
    for (int i = 0; i < n_sequence; ++i) {
        std::span view = std::span(&data[i], window_size);
        auto min = std::numeric_limits<T>::max();
        int min_index = 0;
        for (int j = 0; j < n_sequence; ++j) {
            // Compute Euclidean distance for the current sequence
            const auto distance = euclideanDistance(view, std::span(&data[j], window_size));
            if (i<=3) {
                // printf("%d %d %f\n", i, j, distance*distance);
            }
            if (distance < min and (j < i - exclude or j > i + exclude) ) {
                min = distance;
                min_index = j; 
                // printf("min %d %d %f\n", i, min_index, min*min);
            }
        }
        matrix_profile[i] = min;
        profile_index[i] = min_index; 
        // printf("min %d %d %f\n", i, profile_index[i], matrix_profile[i]*matrix_profile[i]);
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

    std::vector<T> first_row(n_sequence, T(0));
    std::vector<T> row(n_sequence, T(0));
    std::vector<T> oldrow(n_sequence, T(0));
    std::vector<T> nrow(n_sequence, T(0));

    int max_threads_num = omp_get_max_threads();
    std::vector<T> local_min_values(max_threads_num, std::numeric_limits<T>::max());
    std::vector<int> local_mins_indices(max_threads_num, 0);

    #pragma omp parallel shared(first_row, row, nrow, matrix_profile, profile_index, local_min_values, local_mins_indices)
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
            first_row[j] = distance;
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
            // Compute the i-th row of the distance matrix based on the (i-1)-th and the first one (first_row)
            // First chunk
            if (thread_id == 0) {
                nrow[0] = first_row[i];
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
                local_min_values[0] = std::min(local_min_values[0], first_row[i]);
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


template <typename T>
auto computeMatrixProfileSTOMPV2(const std::vector<T> &data,
                               int window_size) {

    const int n_sequence = data.size() - window_size + 1;
    const int exclude = 2;

    std::vector<T> matrix_profile(n_sequence, std::numeric_limits<T>::max());
    std::vector<int> profile_index(n_sequence, 0);
    std::vector<T> first_row(n_sequence, T(0));
    std::vector<std::vector<std::pair<T,int>>> block_min_pair_per_row;
    #pragma omp parallel
    {
        const int threads_num = omp_get_num_threads();
        const int thread_id = omp_get_thread_num();
        const int block_width = n_sequence / threads_num;
        const int block_height = threads_num;
        #pragma omp single
        {
            block_min_pair_per_row.resize(threads_num);
        }
        block_min_pair_per_row[thread_id] = std::vector<std::pair<T,int>>(block_height);
    }
    #pragma omp parallel shared(matrix_profile, profile_index, block_min_pair_per_row, first_row, data)
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
        // lambda functions definitions
        auto update_min = [&](int i, int j, int thread_id, int global_i, int global_j, int exclude, std::pair<T, int>& min_pair) {
            if (row[j] < min_pair.first && (global_j < global_i - exclude || global_j > global_i + exclude)) {
                min_pair.first = row[j];
                min_pair.second = global_j;
            }
        };
        auto update_row = [&](int global_i, int global_j, int prev_int) {
            auto tmp_prev = data[global_j - 1] - data[global_i - 1];
            auto tmp_next = data[global_j + window_size - 1] - data[global_i + window_size - 1];
            return row[prev_int] - tmp_prev*tmp_prev + tmp_next*tmp_next;
        };
        // Compute first row
        std::span view = std::span(&data[0], window_size);
        std::pair<T, int> first_min_pair{std::numeric_limits<T>::max(), -1};
        for (int j=0; j < block_width; ++j) {
            const int global_j = initial_block_start + j;
            const auto distance = dotProduct(view, std::span(&data[global_j], window_size));
            row[j] = distance;   
            update_min(0, j, thread_id, 0, global_j, exclude, first_min_pair);
            first_row[global_j] = distance;  // CACHE INVALIDATION
            //  printf("%d %d %f\n", 0, initial_block_start+ j, distance);
        }
        block_min_pair_per_row[thread_id][0] = first_min_pair;
        
        #pragma omp barrier

        const int last_meta_row_height = (n_sequence%block_height!=0) ? n_sequence%block_height : block_height;
        // For each meta row
        for (int meta_row_id=0; meta_row_id < meta_rows_num; ++meta_row_id) {
            block_id = (meta_row_id + thread_id) % threads_num;
            // Define the block's row start and end: height
            int block_row_start = meta_row_id==0 ? 1 : 0;
            int block_row_end = (meta_row_id==meta_rows_num-1) ? last_meta_row_height : block_height;
            // Define the block's column start and end: width
            int block_column_start;
            if (threads_num == 1) {
                block_column_start = initial_block_start;
            } else {
                block_column_start = (initial_block_start + block_height*meta_row_id) % n_sequence;
            }
            int block_column_end = std::min(block_column_start + block_width, n_sequence);
            // Compute all other rows in block
            for (int i=block_row_start; i < block_row_end; ++i) {
                const int global_i = meta_row_id*block_height + i; // Global row index in the distance matrix
                std::pair<T,int> min_pair{std::numeric_limits<T>::max(), -1};
                // Semi blocks == last block
                if (block_column_end == n_sequence) {
                    int length  = n_sequence - block_column_start;
                    int right_upperbound  = length - i;
                    int left_upperbound = block_width - length + i; 

                    // right semi block
                    int stop = (block_column_start==0) ? 1 : 0;
                    for (int j = right_upperbound-1; j >= stop ; --j) {
                        const int global_j = block_column_start + j + i;
                        int shift = (left_upperbound==0) ? 0 : left_upperbound-1;
                        auto old_row = row[j+shift];

                        row[left_upperbound+j] = update_row(global_i, global_j, shift+j);
                        update_min(i, j, thread_id, global_i, block_column_start + j +i, exclude, min_pair);

                        if (global_i<=3)  printf("R %d %d %d %d %f %f %d\n", global_i, global_j, i, j, row[left_upperbound+j], old_row, thread_id);    
                        if (row[left_upperbound+j] != dotProduct(std::span(&data[global_i], window_size), std::span(&data[global_j], window_size))){
                            printf("Error in right semi block %d %d of length %d and width %d in line %d\n", global_i, global_j, length, block_width, i);
                            printf("Expected %f, got %f\n", dotProduct(std::span(&data[global_i], window_size), std::span(&data[global_j], window_size)), row[left_upperbound+j]);
                            printf("With old %f instead of %f\n", old_row, dotProduct(std::span(&data[global_i-1], window_size), std::span(&data[global_j-1], window_size)));
                            throw std::runtime_error("Error in right semi block\n");
                        }
                    }
                    // left semi block
                    if (block_column_start + block_width >= n_sequence) {    
                        for (int j = left_upperbound-1; j > 0; --j) {
                            int global_j = j;
                            auto old_row = row[j-1];
                            row[j] = update_row(global_i, global_j, j-1);
                            update_min(i, j, thread_id, global_i, global_j, exclude, min_pair);
                            assert(row[j] == dotProduct(std::span(&data[global_i], window_size), std::span(&data[global_j], window_size)));
                            if (global_i<=3) printf("L %d %d %f %f %d \n", global_i, global_j, row[j], old_row, thread_id);    
                        }
                        if (left_upperbound > 0){
                            row[0] = first_row[global_i];
                            update_min(i, 0, thread_id, global_i, 0, exclude, min_pair);
                                assert(row[0] == dotProduct(std::span(&data[global_i], window_size), std::span(&data[0], window_size)));
                            if (global_i<=3)  printf("L %d %d %f %d\n", global_i, 0, row[0], thread_id);    
                        }
                    } else if (block_height == 1) {
                        row[0] = first_row[global_i];
                        update_min(i, 0, thread_id, global_i, 0, exclude, min_pair);
                    }   
                } else {
                    // Default block
                    if (i == 0 and block_column_start == 0) {
                        for (int j = block_width-1; j >=0; --j) {
                            if (j == 0){
                                row[j] = first_row[global_i];
                            } else {
                                const int global_j = block_column_start + i + j;
                                auto old_row = row[j];
                                row[j] = update_row(global_i, global_j, j-1);
                                update_min(i, j, thread_id, global_i, block_column_start + i + j, exclude, min_pair);
                                if (global_i<=3) printf("+ %d %d %f %d\n", global_i, block_column_start + j+i, row[j], thread_id);
                                if  (row[j] != dotProduct(std::span(&data[global_i], window_size), std::span(&data[global_j], window_size))){
                                    printf("+ Error in full block %d %d of width %d in line %d starting at %d \n", global_i, global_j, block_width, i, block_column_start);
                                    printf("Expected %f, got %f\n", dotProduct(std::span(&data[global_i], window_size), std::span(&data[global_j], window_size)), row[j]);
                                    printf("With old %f instead of %f\n", old_row, dotProduct(std::span(&data[global_i-1], window_size), std::span(&data[global_j-1], window_size)));
                                    throw std::runtime_error("Error in full block\n");
                                }
                            }
                        }
                    } else {
                        for (int j = 0; j < block_width; ++j) {
                            const int global_j = block_column_start + i + j;
                            auto old_row = row[j];

                            row[j] =  update_row(global_i, global_j, j);
                            update_min(i, j, thread_id, global_i, block_column_start + i + j, exclude, min_pair);
                            if (global_i<=3) printf("- %d %d %f %d\n", global_i, block_column_start + j+i, row[j], thread_id);


                            if (row[j] != dotProduct(std::span(&data[global_i], window_size), std::span(&data[global_j], window_size))){
                                printf("- Error in full block %d %d of width %d in line %d starting at %d by thread %d \n", global_i, global_j, block_width, i, block_column_start, thread_id);
                                printf("Expected %f, got %f\n", dotProduct(std::span(&data[global_i], window_size), std::span(&data[global_j], window_size)), row[j]);
                                printf("With old %f instead of %f\n", old_row, dotProduct(std::span(&data[global_i-1], window_size), std::span(&data[global_j-1], window_size)));
                                throw std::runtime_error("Error in full block\n");
                            }
                        }
                    }
                }
            
            block_min_pair_per_row[thread_id][i] = min_pair;
            #pragma omp barrier
            }
            #pragma omp barrier
            // Compute global minimum per row
            if (thread_id < block_row_end) {
                auto min = std::numeric_limits<T>::max();
                int ind = -1;
                for (int k = 0; k < threads_num; ++k) {
                    auto min_pair = block_min_pair_per_row[k][thread_id].first;
                    if (min_pair < min) {
                        min = min_pair;
                        ind = block_min_pair_per_row[k][thread_id].second;
                    }
                }
                matrix_profile[meta_row_id*block_height + thread_id] = std::sqrt(std::abs(min)); // abs to fix min=-0,0000  
                profile_index[meta_row_id*block_height + thread_id] = ind; 
                // printf("min %d %d %f\n", meta_row_id*block_height + thread_id, profile_index[meta_row_id*block_height + thread_id], matrix_profile[meta_row_id*block_height + thread_id]*matrix_profile[meta_row_id*block_height + thread_id]);
            }
            #pragma omp barrier
        }
    }
    return std::make_tuple(matrix_profile, profile_index);
}


#endif // MATRIX_PROFILE_H