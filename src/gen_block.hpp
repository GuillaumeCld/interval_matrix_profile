#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <functional>
#include <span>
#include <distance.hpp>
#include <cfloat>

/**
 * @file block.hpp
 */
template <typename T>
class gblock
{
public:
    using value_type = T;
    using pair_t = min_pair<T>;
    std::function<void(gblock<T> *)> compute_method; // the pointer to the compute function
    int ID;                                          // ID of the block
    gblock() = default;                              // Default constructor

    /**
     * @brief Constructor
     */
    gblock(const int n,
           const int m,
           const int i,
           const int j,
           const int ID,
           const int width,
           const int height,
           std::span<T> const in_query_first_row,
           std::span<T> const in_target_first_row,
           std::span<T> const in_query_time_series,
           std::span<T> const in_target_time_series,
           std::vector<min_pair<T>> &in_local_min_row)
        : _n(n),
          _m(m),
          _global_i(i),
          _global_j(j),
          ID(ID),
          _width(width),
          _height(height),
          query_first_row(in_query_first_row),
          target_first_row(in_target_first_row),
          query_time_series(in_query_time_series),
          target_time_series(in_target_time_series),
          local_min_row(in_local_min_row)
    {
        this->row.resize(width);
        // Case parallelogram truncated on the left
        if (j < 0) [[unlikely]]
        {
            if (j + _width <= 0 and j + _height <= 0)
            {
                compute_method = &gblock<T>::compute_triangle;
                _type = TRIANGLE;
            }
            else if (j + _width > 0 and j + _height <= 0)
            {
                compute_method = &gblock<T>::compute_quadrangle_with_initial_recurrence;
                _type = QUADRANGLE_WITH_INITIAL_RECURRENCE;
            }
            else if (j + width <= 0 and j + height > 0)
            {
                compute_method = &gblock<T>::compute_quadrangle_without_initial_recurrence;
                _type = QUADRANGLE_WITHOUT_INITIAL_RECURRENCE;
            }
            else if (j + _width > 0 and j + _height > 0)
            {
                compute_method = &gblock<T>::compute_polygon;
                _type = POLYGON;
            }
        }
        else if (j + _width + _height > n) [[unlikely]]
        {
            compute_method = &gblock<T>::compute_right_truncated_parallelogram;
            _type = RIGHT_TRUNCATED_PARALLELOGRAM;
        }
        else [[likely]]
        {
            compute_method = &gblock<T>::compute_parallelogram;
            _type = PARALLELOGRAM;
        }
    }
    /**
     * @brief Destructor
     */
    ~gblock()
    {
        // Destructor
    }

    /**
     * @brief Get the row needed for the recurrence in the block
     */
    std::vector<T> &get_row()
    {
        return this->row;
    }

    /**
     * @brief Get the array of minimum per row in the block
     */
    std::vector<min_pair<T>> &get_local_min_rows()
    {
        return this->local_min_row;
    }

    /**
     * @brief Get the type of the block
     */
    int get_type()
    {
        return _type;
    }

    /**
     * @brief Compute the minimum per row in the block using the compute procedure
     */
    inline void compute()
    {
        compute_method(this);
    }

    void print(std::ostream &out)
    {
        out << "Block " << " type " << _type << " i " << _global_i << " j " << _global_j << " height " << _height << " width " << _width << std::endl;
        out << "Query first row: ";
        for (int i = 0; i < _width; ++i)
        {
            out << this->query_first_row[i] << " ";
        }
        out << std::endl;
    }

private:
    int _type;                              // type of the block
    int _n;                                 // length of the distance matrix (the number of sequences )
    int _m;                                 // the window size
    int _width;                             // width
    int _height;                            // height
    int _global_i;                          // i coordinate top left corner
    int _global_j;                          // j coordinate top left corner
    std::vector<T> row;                     // the row for the recurrence in the block
    std::span<T> query_first_row;           // the first row of the distance matrix for the query
    std::span<T> target_first_row;          // the first row of the distance matrix for the target
    std::span<T> query_time_series;         // the query time series
    std::span<T> target_time_series;        // the target time series
    std::vector<min_pair<T>> local_min_row; // the array of minimum per row in the block

    /**
     * @brief Update the minimum value and index of the pair
     * @param index the index of the minimum value to compare
     * @param min the pair to update
     */
    inline void update_min(const int index, min_pair<T> &min, const int global_i, const int global_j)
    {
        if (this->row[index] < min.value)
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
    inline void update_row_element(const int j, const int global_i, const int global_j)
    {
        // Compute the elements to remove (prev) and the elements to add (next)
        const T prev_data{this->query_time_series[global_i - 1] - this->target_time_series[global_j - 1]};
        const T next_data{this->query_time_series[global_i + _m - 1] - this->target_time_series[global_j + _m - 1]};
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
    inline void compute_row(const int start, const int end, int i, min_pair<T> &min)
    {
        const int global_i = _global_i + i;
        int global_j{_global_j + start + i};
        // splited in two loops for vectorization purposes
        for (int j = start; j < end; ++j)
        {
            const T prev_data{this->query_time_series[global_i - 1] - this->target_time_series[global_j - 1]};
            const T next_data{this->query_time_series[global_i + _m - 1] - this->target_time_series[global_j + _m - 1]};
            // Update the row following the recurrence
            this->row[j] += (next_data * next_data - prev_data * prev_data);
            ++global_j;
        }
        global_j = _global_j + start + i;
        for (int j = start; j < end; ++j)
        {
            update_min(j, min, global_i, global_j);
            ++global_j;
        }
        this->local_min_row[i] = min;
    }

    /**
     * @brief Update the row with brute force and compute the minimum of the row.
     *
     * @param start The start index of the row.
     * @param end The end index of the row.
     * @param i The index of the row.
     * @param min The pair to update.
     */
    inline void compute_row_brute_force(const int start, const int end, int i, min_pair<T> &min)
    {
        const int global_i = _global_i + i;
        int global_j = _global_j + start + i;
        std::span<T> view = std::span(&query_time_series[global_i], _m);
        for (int j = start; j < end; ++j)
        {
            update_row_element(j, global_i, global_j);
            this->row[j] = dotProduct(view, std::span(&target_time_series[global_j], _m));
            update_min(j, min, global_i, global_j);
            ++global_j;
        }
        this->local_min_row[i] = min;
    }

    /**
     * @brief Initialize the block with the first row of the distance matrix and compute the minimum
     * @param start the start index of the row
     * @param end the end index of the row
     * @param min the pair to update
     */
    inline void initialize_with_first_row(const int start, const int end, min_pair<T> &min)
    {
        for (int j = start; j < end; ++j)
        {
            this->row[j] = this->target_first_row[_global_j + j];
            update_min(j, min, _global_i, _global_j + j);
        }
        this->local_min_row[0] = min;
    }


    /**
     * @brief Compute the minimum per row in the block in the parallelogram case
     */
    inline void compute_parallelogram()
    {
        min_pair<T> min = this->local_min_row[0];
        // Initialize the first row
        if (_global_i == 0) [[unlikely]]
        {
            // Case first row of the distance matrix
            initialize_with_first_row(0, _width, min);
        }
        else [[likely]]
        {
            const int i{0};
            if (_global_j == 0) [[unlikely]]
            {
                // Case first column of the distance matrix
                this->row[0] = this->query_first_row[_global_i];
                update_min(0, min, _global_i, 0);
                const int j{1};

                compute_row_brute_force(j, _width, i, min);
                
            }
            else [[likely]]
            {
                const int j{0};

                compute_row_brute_force(j, _width, i, min);
                
            }
        }
        const int j{0};
        for (int i = 1; i < _height; ++i)
        {
            min_pair<T> min = this->local_min_row[i];
            compute_row(j, _width, i, min);
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the triangle case
     */
    inline void compute_triangle()
    {
        const int first_line = -(_global_j + _width) + 1;
        const int elem_per_row{1};
        int current_start{_width - elem_per_row};
        for (int i = first_line; i < _height; ++i)
        {
            min_pair<T> min = this->local_min_row[i];
            // Compute the first element of the row after the truncation

            this->row[current_start] = this->query_first_row[_global_i + i];
            update_min(current_start, min, _global_i + i, 0);
            // Compute the rest of the row
            compute_row(current_start + 1, _width, i, min);
            --current_start;
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the polygon case
     */
    inline void compute_polygon()
    {

        const int nb_left_elements{-_global_j};
        const int nb_right_elements{_width - nb_left_elements};
        min_pair<T> min = this->local_min_row[0];
        // intialize the first row
        this->row[nb_left_elements] = this->query_first_row[_global_i];
        update_min(nb_left_elements, min, _global_i, 0);

        int global_j{_global_j + nb_left_elements + 1};

        if (_global_i == 0)
        {
            for (int j = 1; j < nb_right_elements; ++j)
            {
                this->row[nb_left_elements + j] = this->target_first_row[_global_i + j];
                update_min(nb_left_elements + j, min, _global_i, j);
            }
        }
        else
        {
            std::span view = std::span(&query_time_series[_global_i], _m);
            for (int j = 1; j < nb_right_elements; ++j)
            {
                this->row[nb_left_elements + j] = dotProduct(view, std::span(&target_time_series[global_j], _m));
                update_min(nb_left_elements + j, min, _global_i, global_j);
                ++global_j;
            }
        
        }
        this->local_min_row[0] = min;
        // Compute the first truncated rows
        // |-------
        // |       \
        // |        \
        // |---------|
        //
        int global_i{_global_i + 1};
        int row_start{nb_left_elements - 1};
        for (int i = 1; i < nb_left_elements + 1; ++i)
        {
            min_pair<T> min = this->local_min_row[i];
            // Compute the first element of the row after the truncation
            this->row[row_start] = this->query_first_row[global_i];
            update_min(row_start, min, global_i, 0);
            // Compute the rest of the row
            compute_row(row_start + 1, _width, i, min);
            ++global_i;
            --row_start;
        }
        // Compute the rest of the parallelogram
        for (int i = nb_left_elements + 1; i < _height; ++i)
        {
            min_pair<T> min = this->local_min_row[i];
            compute_row(0, _width, i, min);
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the quadrangle case without initial recurrence
     */
    inline void compute_quadrangle_without_initial_recurrence()
    {
        const int first_line{1 - (_global_j + _width)};
        // Triangle part
        const int elem_per_row{1};
        int current_start{_width - elem_per_row};
        int global_i{_global_i + first_line};
        for (int i = first_line; i < first_line + _width; ++i)
        {
            min_pair<T> min = this->local_min_row[i];
            // First element of the row after the truncation
            this->row[current_start] = this->query_first_row[global_i];
            update_min(current_start, min, global_i, 0);
            // Compute the rest of the row
            compute_row(current_start + 1, _width, i, min);
            ++global_i;
            --current_start;
        }
        // Parallelogram part
        for (int i = first_line + _width; i < _height; ++i)
        {
            min_pair<T> min = this->local_min_row[i];
            compute_row(0, _width, i, min);
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the quadrangle case with initial recurrence
     */
    inline void compute_quadrangle_with_initial_recurrence()
    {
        const int elem_per_row{_width + _global_j};
        int current_start{_width - elem_per_row};
        min_pair<T> min = this->local_min_row[0];
        // Initialize the first row
        // compute the first element of the row
        this->row[current_start] = this->query_first_row[_global_i];
        update_min(current_start, min, _global_i, 0);
        // Compute the rest of the row

        compute_row_brute_force(current_start + 1, _width, 0, min);
        
        --current_start;
        int global_i{_global_i + 1};
        // Compute the rest of the quadrangle
        for (int i = 1; i < _height; ++i)
        {
            min_pair<T> min = this->local_min_row[i];
            // Compute the first element of the row
            this->row[current_start] = this->query_first_row[global_i];
            update_min(current_start, min, global_i, 0);
            // Compute the rest of the row
            compute_row(current_start + 1, _width, i, min);
            ++global_i;
            --current_start;
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the right truncated parallelogram case
     */
    inline void compute_right_truncated_parallelogram()
    {
        const int i_max{std::min(_n - _global_i, _height)};
        int j_max{std::min(_n - _global_j, _width)};
        min_pair<T> min = this->local_min_row[0];
        if (_global_i == 0)
        {
            initialize_with_first_row(0, j_max, min);
        }
        else
        {

            compute_row_brute_force(0, j_max, 0, min);
            
        }
        for (int i = 1; i < i_max; ++i)
        {
            min_pair<T> min = this->local_min_row[i];
            j_max = std::min(_n - (_global_j + i), _width);
            compute_row(0, j_max, i, min);
        }
    }
};