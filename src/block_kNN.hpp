#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <functional>
#include <distance.hpp>
#include <cfloat>
#include "block.hpp"

/**
 * @file block.hpp
 */
template <typename T, typename HeapType>
class block_kNN
{
public:
    using value_type = T;
    using heap_type = HeapType;
    using pair_type = HeapType::value_type;
    std::function<void(block_kNN<T, HeapType> *)>
        STOMP_method;      // the pointer to the STOMP function
    int ID;                // ID of the block
    block_kNN() = default; // Default constructor

    block_kNN(const int n,
              const int m,
              const int exclude,
              const int i,
              const int j,
              const int ID,
              const int width,
              const int height,
              const int k,
              std::span<value_type> const in_first_row,
              std::span<value_type> const in_initial_row,
              std::span<value_type> const in_time_series,
              std::vector<heap_type> *in_heap_per_row)
        : _n(n),
          _m(m),
          _exclude(exclude),
          _global_i(i),
          _global_j(j),
          ID(ID),
          _width(width),
          _height(height),
          _k(k),
          _max_heap_size(k * (exclude * 2)),
          first_row(in_first_row),
          initial_row(in_initial_row),
          time_series(in_time_series),
          heap_per_row(in_heap_per_row)
    {
        this->row.resize(width);
        // Case parallelogram truncated on the left
        if (j < 0) [[unlikely]]
        {
            if (j + _width <= 0 and j + _height <= 0)
            {
                STOMP_method = &block_kNN<T, HeapType>::STOMP_triangle;
                _type = TRIANGLE;
            }
            else if (j + _width > 0 and j + _height <= 0)
            {
                STOMP_method = &block_kNN<T, HeapType>::STOMP_quadrangle_with_initial_recurrence;
                _type = QUADRANGLE_WITH_INITIAL_RECURRENCE;
            }
            else if (j + width <= 0 and j + height > 0)
            {
                STOMP_method = &block_kNN<T, HeapType>::STOMP_quadrangle_without_initial_recurrence;
                _type = QUADRANGLE_WITHOUT_INITIAL_RECURRENCE;
            }
            else if (j + _width > 0 and j + _height > 0)
            {
                STOMP_method = &block_kNN<T, HeapType>::STOMP_polygon;
                _type = POLYGON;
            }
        }
        else if (j + _width + _height > n) [[unlikely]]
        {
            STOMP_method = &block_kNN<T, HeapType>::STOMP_right_truncated_parallelogram;
            _type = RIGHT_TRUNCATED_PARALLELOGRAM;
        }
        else [[likely]]
        {
            STOMP_method = &block_kNN<T, HeapType>::STOMP_parallelogram;
            _type = PARALLELOGRAM;
        }
    }
    /**
     * @brief Destructor
     */
    ~block_kNN()
    {
        // Destructor
    }

    /**
     * @brief Get the row needed for the recurrence in the block
     */
    std::vector<value_type> &get_row()
    {
        return this->row;
    }

    /**
     * @brief Get the array of minimum per row in the block
     */
    std::vector<heap_type> &get_heap_per_row()
    {
        return this->heap_per_row;
    }

    /**
     * @brief Get the type of the block
     */
    int get_type()
    {
        return _type;
    }

    /**
     * @brief Compute the minimum per row in the block using the STOMP procedure
     */
    inline void STOMP()
    {
        STOMP_method(this);
    }

    void print(std::ostream &out)
    {
        out << "Block " << " type " << _type << " i " << _global_i << " j " << _global_j << " height " << _height << " width " << _width << std::endl;
        out << "Initial row: ";
        for (int i = 0; i < _width; ++i)
        {
            out << this->initial_row[i] << " ";
        }
        out << std::endl;
        // out << "Row: ";
        // for (int i = 0; i < _width; ++i)
        // {
        //     out << this->row[i] << " ";
        // }
        out << std::endl;
    }

private:
    int _type;                            // type of the block
    int _n;                               // length of the distance matrix (the number of sequences )
    int _m;                               // the window size
    int _width;                           // width
    int _height;                          // height
    int _exclude;                         // exclusion zone
    int _global_i;                        // i coordinate top left corner
    int _global_j;                        // j coordinate top left corner
    int _k;                               // number of nearest neighor to look for
    int _max_heap_size;                   // maximum size of the heap (k * exclude)
    std::vector<value_type> row;          // the row for the recurrence in the block
    std::span<value_type> first_row;      // the first row of the distance matrix
    std::span<value_type> initial_row;    // the initial row for the recurrence
    std::span<value_type> time_series;    // the time series
    std::vector<heap_type> *heap_per_row; // the array of minimum per row in the block

    /**
     * @brief Update the minimum value and index of the pair
     * @param index the index of the minimum value to compare
     * @param min the pair to update
     */
    inline void update_min(const int index, const int row, const int global_i, const int global_j)
    {
        const value_type value{this->row[index]};   
        heap_type& heap = (*heap_per_row)[row];
        if (heap.size() == _max_heap_size)
        {
            if (value < heap.top().value and std::abs(global_j - global_i) > _exclude)
            {
                heap.pop();
                heap.push({global_j, value});
            }
        }
        else
        {
            if (std::abs(global_j - global_i) > _exclude)
            {
                heap.push({global_j, value});
            }
        }
    }

    /**
     * @brief Update the row for the recurrence in the block
     * @param j the index of the element to update in the row
     * @param _global_i the global i coordinate
     * @param _global_j the global j coordinate
     */
    inline void update_row(const int j, const int global_i, const int global_j)
    {
        // Compute the elements to remove (prev) and the elements to add (next)
        const value_type prev_data{this->time_series[global_i - 1] - this->time_series[global_j - 1]};
        const value_type next_data{this->time_series[global_i + _m - 1] - this->time_series[global_j + _m - 1]};
        // Update the row following the recurrence
        this->row[j] += (next_data * next_data - prev_data * prev_data);
    }

    /**
     * @brief Update the recurrence of the row and compute the minimum of the row.
     *
     * @param start The start index of the row.
     * @param end The end index of the row.
     * @param i The index of the row.
     * @param min The pair to update.
     */
    inline void compute_row(const int start, const int end, int i)
    {
        const int global_i = _global_i + i;
        int global_j = _global_j + start + i;
        for (int j = start; j < end; ++j)
        {
            update_row(j, global_i, global_j);
            update_min(j, i, global_i, global_j);
            ++global_j;
        }
    }

    /**
     * @brief Initialize the block with the first row of the distance matrix and compute the minimum
     * @param start the start index of the row
     * @param end the end index of the row
     * @param min the pair to update
     */
    inline void initialize_with_first_row(const int start, const int end)
    {
        for (int j = start; j < end; ++j)
        {
            this->row[j] = this->first_row[_global_j + j];
            update_min(j, 0, _global_i, _global_j + j);
        }
    }

    /**
     * @brief Initialize the block with the initial row for the recurrence and compute the minimum
     * @param start the start index of the row
     * @param end the end index of the row
     * @param min the pair to update
     */
    inline void initialize_with_initial_row(const int start, const int end)
    {
        int global_j = _global_j + start;
        for (int j = start; j < end; ++j)
        {
            // Compute the elements to remove (prev) and the elements to add (next)
            const value_type prev_data{this->time_series[global_j - 1] - this->time_series[_global_i - 1]};
            const value_type next_data{this->time_series[global_j + _m - 1] - this->time_series[_global_i + _m - 1]};
            // Update the row following the recurrence
            this->row[j] = this->initial_row[j] + (next_data * next_data - prev_data * prev_data);
            update_min(j, 0, _global_i, global_j);
            ++global_j;
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the parallelogram case
     */
    inline void STOMP_parallelogram()
    {
        // Initialize the first row
        if (_global_i == 0) [[unlikely]]
        {
            // Case first row of the distance matrix
            initialize_with_first_row(0, _width);
        }
        else [[likely]]
        {
            // Case initialize with the given initial row
            if (_global_j == 0) [[unlikely]]
            {
                // Case first column of the distance matrix
                this->row[0] = this->first_row[_global_i];
                update_min(0, 0, _global_i, _global_j);
                initialize_with_initial_row(1, _width);
            }
            else [[likely]]
            {
                initialize_with_initial_row(0, _width);
            }
        }
        // Compute the other rows
        for (int i = 1; i < _height; ++i)
        {
            compute_row(0, _width, i);
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the triangle case
     */
    inline void STOMP_triangle()
    {
        const int first_line{-(_global_j + _width) + 1};
        int elem_per_row{1};
        int current_start{_width - elem_per_row};
        int global_i{_global_i + first_line};
        for (int i = first_line; i < _height; ++i)
        {
            // Compute the first element of the row after the truncation
            this->row[current_start] = this->first_row[global_i];
            update_min(current_start, i, global_i, 0);
            // Compute the rest of the row
            compute_row(current_start + 1, _width, i);
            --current_start;
            ++global_i;
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the polygon case
     */
    inline void STOMP_polygon()
    {

        const int nb_left_elements{-_global_j};
        const int nb_right_elements{_width - nb_left_elements};

        // intialize the first row
        this->row[nb_left_elements] = this->first_row[_global_i];
        update_min(nb_left_elements, 0, _global_i, 0);

        int global_j{_global_j + nb_left_elements + 1};

        if (_global_i == 0)
        {
            for (int j = 1; j < nb_right_elements; ++j)
            {
                this->row[nb_left_elements + j] = this->first_row[_global_i + j];
                update_min(nb_left_elements + j, 0, _global_i, j);
            }
        }
        else
        {
            for (int j = 1; j < nb_right_elements; ++j)
            {
                const value_type prev_data{this->time_series[global_j - 1] - this->time_series[_global_i - 1]};
                const value_type next_data{this->time_series[global_j + _m - 1] - this->time_series[_global_i + _m - 1]};
                this->row[nb_left_elements + j] = this->initial_row[nb_left_elements + j] + (next_data * next_data - prev_data * prev_data);
                update_min(nb_left_elements + j, 0, _global_i, global_j);
                ++global_j;
            }
        }

        // Compute the first truncated rows
        int global_i{_global_i + 1};
        int row_start{nb_left_elements - 1};
        for (int i = 1; i < nb_left_elements + 1; ++i)
        {
            // Compute the first element of the row after the truncation
            this->row[row_start] = this->first_row[global_i];
            update_min(row_start, i, global_i, 0);
            // Compute the rest of the row
            compute_row(row_start + 1, _width, i);
            ++global_i;
            --row_start;
        }
        // Compute the rest of the parallelogram
        for (int i = nb_left_elements + 1; i < _height; ++i)
        {
            compute_row(0, _width, i);
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the quadrangle case without initial recurrence
     */
    inline void STOMP_quadrangle_without_initial_recurrence()
    {
        const int first_line{-(_global_j + _width) + 1};
        // Triangle part
        int elem_per_row{1};
        int current_start{_width - elem_per_row};
        int global_i{_global_i + first_line};
        for (int i = first_line; i < first_line + _width; ++i)
        {
            // First element of the row after the truncation
            this->row[current_start] = this->first_row[global_i];
            update_min(current_start, i, global_i, 0);
            // Compute the rest of the row
            compute_row(current_start + 1, _width, i);
            ++global_i;
            --current_start;
        }
        // Parallelogram part
        for (int i = first_line + _width; i < _height; ++i)
        {
            compute_row(0, _width, i);
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the quadrangle case with initial recurrence
     */
    inline void STOMP_quadrangle_with_initial_recurrence()
    {
        const int elem_per_row{_width + _global_j};
        int current_start{_width - elem_per_row};
        // Initialize the first row
        this->row[current_start] = this->first_row[_global_i];
        update_min(current_start, 0, _global_i, 0);

        int global_j{_global_j + current_start + 1};
        for (int j = current_start + 1; j < _width; ++j)
        {
            // Compute the elements to remove (prev) and the elements to add (next)
            const value_type prev_data{this->time_series[global_j - 1] - this->time_series[_global_i - 1]};
            const value_type next_data{this->time_series[global_j + _m - 1] - this->time_series[_global_i + _m - 1]};
            // Update the row following the recurrence
            this->row[j] = this->initial_row[j] + (next_data * next_data - prev_data * prev_data);
            update_min(j, 0, _global_i, global_j);
            ++global_j;
        }
        --current_start;
        int global_i{_global_i + 1};
        // Compute the rest of the quadrangle
        for (int i = 1; i < _height; ++i)
        {
            // Compute the first element of the row
            this->row[current_start] = this->first_row[global_i];
            update_min(current_start, i, global_i, 0);
            // Compute the rest of the row
            compute_row(current_start + 1, _width, i);
            ++global_i;
            --current_start;
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the right truncated parallelogram case
     */
    inline void STOMP_right_truncated_parallelogram()
    {
        const int i_max{std::min(_n - _global_i, _height)};
        int j_max{std::min(_n - _global_j, _width)};

        // Initialize the first row
        if (_global_i > 0)
        {
            initialize_with_initial_row(0, j_max);
        }
        else
        {
            initialize_with_first_row(0, j_max);
        }
        // Compute the other rows
        for (int i = 1; i < i_max; ++i)
        {
            j_max = std::min(_n - (_global_j + i), _width);
            compute_row(0, j_max, i);
        }
    }
};