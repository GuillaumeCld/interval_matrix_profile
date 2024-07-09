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

/**
 * @brief A class representing a square block.
 *
 * @tparam T The type of elements in the block.
 */
template <typename T>
class square_block
{
public:
    square_block() = default; // Default constructor

    /**
     * @brief Constructor for the square block.
     *
     * @param n The length of the distance matrix (the number of sequences).
     * @param m The window size.
     * @param exclude The exclusion zone.
     * @param i The i coordinate of the top left corner.
     * @param j The j coordinate of the top left corner.
     * @param width The width of the block.
     * @param height The height of the block.
     * @param in_local_min_row The array of minimum per row in the block.
     * @param in_time_series The time series.
     */
    square_block(const int n,
                 const int m,
                 const int exclude,
                 const int i,
                 const int j,
                 const int width,
                 const int height,
                 std::vector<min_pair<T>> & in_local_min_row,
                 std::span<T> in_time_series)
        : _n(n),
          _m(m),
          _exclude(exclude),
          _global_i(i),
          _global_j(j),
          _width(width),
          _height(height),
          local_min_row(in_local_min_row),
          time_series(in_time_series)
    {
    }

    /**
     * @brief Destructor for the square block.
     */
    ~square_block()
    {
        // Destructor
    }

    /**
     * @brief Get the array of minimum per row in the block.
     *
     * @return std::vector<min_pair<T>>& The array of minimum per row.
     */
    std::vector<min_pair<T>> get_local_min_rows()
    {
        return this->local_min_row;
    }

    /**
     * @brief Compute the minimum per row in the block using the STOMP procedure.
     */
    inline void STOMP()
    {
        std::vector<T> row(_width, 0);
        int global_i = _global_i;
        int global_j = _global_j;
        const std::span view = std::span(&this->time_series[global_i], _m);
        min_pair<T> min{this->local_min_row[0]};
        for (int j = 0; j < _width; ++j)
        {
            const T distance = dotProduct(view, std::span(&this->time_series[global_j], _m));
            row[j] = distance;
            if (distance < min.value and (global_j < global_i - _exclude or global_j > global_i + _exclude))
            {
                min.value = distance;
                min.index = global_j;
            }
            ++global_j;
        }
        this->local_min_row[0] = min;
        ++global_i;

        for (int i = 1; i < _height; ++i)
        {
            global_j = _global_j + _width - 1;
            min_pair<T> min{this->local_min_row[i]};

            for (int j = _width - 1; j > 0; --j)
            {
                const T prev_data = this->time_series[global_i - 1] - this->time_series[global_j - 1];
                const T next_data = this->time_series[global_i + _m - 1] - this->time_series[global_j + _m - 1];
                const T distance = row[j - 1] + (next_data * next_data - prev_data * prev_data);
                row[j] = distance;
                if (distance < min.value and (global_j < global_i - _exclude or global_j > global_i + _exclude))
                {
                    min.value = distance;
                    min.index = global_j;
                }
                --global_j;
            }
            const T distance = dotProduct(std::span(&this->time_series[global_i], _m), std::span(&this->time_series[_global_j], _m));
            row[0] = distance;
            if (distance < min.value and (_global_j < global_i - _exclude or _global_j > global_i + _exclude))
            {
                min.value = distance;
                min.index = _global_j;
            }
            ++global_i;
            // Write the row minimum to the local_min_row
            this->local_min_row[i] = min;
        }
    }

    /**
     * @brief Compute the minimum per row in the block using brute force.
     */
    inline void brute_force()
    {
        int global_i = _global_i;
        for (int i = 0; i < _height; ++i)
        {
            min_pair<T> min{this->local_min_row[i]};
            const std::span view = std::span(&this->time_series[global_i], _m);

            int global_j = _global_j;
            for (int j = 0; j < _width; ++j)
            {
                const T distance = dotProduct(view, std::span(&this->time_series[global_j], _m));
                if (distance < min.value and (global_j < global_i - _exclude or global_j > global_i + _exclude))
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

    /**
     * @brief Print the block to the specified output stream.
     *
     * @param out The output stream to print to.
     */
    void print(std::ostream &out)
    {
        out << "Block " << " i " << _global_i << " j " << _global_j << std::endl;
        out << "Min row: ";
        for (int i = 0; i < _height; ++i)
        {
            out << this->local_min_row[i] << " ";
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
    std::span<T> time_series;               // the time series
    std::vector<min_pair<T>> local_min_row; // the array of minimum per row in the block
};