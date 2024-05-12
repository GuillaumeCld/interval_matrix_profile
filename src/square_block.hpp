#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <functional>
#include <distance.hpp>
#include <cfloat>
#include "block.hpp"
#include <span>

/**
 * @file block.hpp
 */
template <typename T>
class square_block
{
public:
    int ID;            // ID of the block
    square_block() = default; // Default constructor

    /**
     * @brief Constructor
     */
    square_block(int n,
          int m,
          int exclude,
          int i,
          int j,
          int ID,
          int width,
          int height,
          std::vector<T> const &in_first_row,
          std::vector<T> &local_min_row,
          std::vector<T> const &in_time_series)
        : _n(n),
          _m(m),
          _exclude(exclude),
          _global_i(i),
          _global_j(j),
          ID(ID),
          _width(width),
          _height(height),
          first_row(in_first_row),
          local_min_row(local_min_row),
          time_series(in_time_series)
    {
    }
    /**
     * @brief Destructor
     */
    ~block()
    {
        // Destructor
    }
    /**
     * @brief Get the array of minimum per row in the block
     */
    std::vector<min_pair<T>> get_local_min_rows()
    {
        return this->local_min_row;
    }

    /**
     * @brief Compute the minimum per row in the block using the STOMP procedure
     */
    inline void STOMP()
    {
        std::vector<T> row(_width, 0);
        int global_i = _global_i;
        int global_j = _global_j;
        std::span<T> view = std::span(&this->time_series[global_i], _m);

        for (int j = 0; j < _width; ++j)
        {
            row[j] = dot_product(view, std::span(&this->time_series[global_j], _m));
            update_min(j, min, global_i, global_j);
            ++global_j;
        }
        this->local_min_row[0] = min;
        ++global_i;

        
        for (int i = 1; i < _height; ++i)
        {
            global_j = _global_j + _width - 1;
            min_pair<T> min{-1, std::numeric_limits<T>::max()};

            for (int j = _width - 1; j > 0; --j)
            {
                const T prev_data = this->time_series[global_i - 1] - this->time_series[global_j - 1];
                const T next_data = this->time_series[global_i + _m - 1] - this->time_series[global_j + _m - 1];
                row[j] = row[j-1] + (next_data * next_data - prev_data * prev_data);
                update_min(j, min, global_i, global_j);
                --global_j;
            }
            row[0] = dot_product(std::span(&this->time_series[global_i], _m), std::span(&this->time_series[global_j], _m));
            ++global_i;
            update_min(0, min, global_i, global_j);
            this->local_min_row[i] = min;
        }
    }

    inline void brute_force()
    {
        int global_i = _global_i;
        for (int i = 0; i < _height; ++i)
        {
            min_pair<T> min{-1, std::numeric_limits<T>::max()};
            std::span<T> view = std::span(&this->time_series[global_i], _m);
            int global_j = _global_j;

            for (int j = 0; j < _width; ++j)
            {
                const T distance = dot_product(view, std::span(&this->time_series[global_j], _m));
                if (distance < min.value)
                {
                    min.value = distance;
                    min.index = global_j;
                }
                ++global_j;
            }
            this->local_min_row[i] = min;
            ++global_i;
        }
    }

    void print(std::ostream &out)
    {
        out << "Block " << " i " << _global_i << " j " << _global_j << std::endl;
        out << "Initial row: ";
        for (int i = 0; i < _width; ++i)
        {
            out << this->initial_row[i] << " ";
        }
        out << std::endl;
        out << "Row: ";
        for (int i = 0; i < _width; ++i)
        {
            out << this->row[i] << " ";
        }
        out << std::endl;
    }

private:
    int _n;                                 // length of the distance matrix (the number of sequences )
    int _m;                                 // the window size
    int _width;                             // width
    int _height;                            // height
    int _exclude;                           // exclusion zone
    int _global_i;                          // i coordinate top left corner
    int _global_j;                          // j coordinate top left corner
    std::vector<T> first_row;               // the first row of the distance matrix
    std::vector<T> time_series;             // the time series
    std::vector<min_pair<T>> local_min_row; // the array of minimum per row in the block

    /**
     * @brief Update the minimum value and index of the pair
     * @param index the index of the minimum value to compare
     * @param min the pair to update
     */
    inline void update_min(int index, min_pair<T> &min, int global_i, int global_j)
    {
        if (this->row[index] < min.value and (global_j < global_i - _exclude or global_j > global_i + _exclude))
        {
            min.value = this->row[index];
            min.index = global_j;
        }
    }

    /**
     * @brief Update the row for the recurrence in the block
     * @param j the index of the element to update in the row
     * @param _global_i the global i coordinate
     * @param _global_j the global j coordinate
     */
    inline void update_row(int j, int global_i, int global_j)
    {
        // Compute the elements to remove (prev) and the elements to add (next)
        auto prev_data = this->time_series[global_i - 1] - this->time_series[global_j - 1];
        auto next_data = this->time_series[global_i + _m - 1] - this->time_series[global_j + _m - 1];
        // Update the row following the recurrence
        this->row[j] += (next_data * next_data - prev_data * prev_data);
    }

    /**
     * @brief Update the recurrence of the row and compute the minimum of the row
     * @param start the start index of the row
     * @param end the end index of the row
     * @param i the index of the row
     * @param min the pair to update
     */
    inline void compute_row(int start, int end, int i, min_pair<T> &min)
    {
        for (int j = start; j < end; ++j)
        {
            update_row(j, _global_i + i, _global_j + i + j);
            update_min(j, min, _global_i + i, _global_j + j + i);
        }
        this->local_min_row[i] = min;
    }
};