#include <vector>
#include <limits>
#include <functional>

enum block_type
{
    PARALLELOGRAM,
    TRIANGLE,
    POLYGON,
    QUADRANGLE_WITHOUT_INITIAL_RECURRENCE,
    QUADRANGLE_WITH_INITIAL_RECURRENCE,
    RIGHT_TRUNCATED_PARALLELOGRAM
};

template <typename T>
struct min_pair
{
    int index;
    T value;
};

/**
 * @file block.hpp
 */
template <typename T>
class block
{
public:
    std::function<void()> STOMP; // the pointer to the STOMP function
    const int ID;                // ID of the block

    /**
     * @brief Constructor
     */
    block(int n, int m, int exclude, int i, int j, int ID, int width, int height, std::vector<T> &first_row, std::vector<T> &initial_row, std::vector<T> &time_series)
        : n(n), m(m), global_i(i), global_j(j), ID(ID), width(width), height(height)
    {

        this.local_min_row.resize(width);
        this.row.resize(width);
        this.first_row = first_row;
        this.initial_row = initial_row;
        this.time_series = time_series;
        if (i < 0)
        {
            // Case parallelogram truncated on the left
            if (i + width <= 0 and i + height <= 0)
            {
                // Case triangle
                this.STOMP = this.STOMP_triangle;
                this.type = TRIANGLE;
            }
            else if (i + width > 0 and i + height <= 0)
            {
                // Case quadrangle with initial recurrence
                // --------
                // |       \
                // |        \
                // |         \
                // ------------
                this.STOMP = this.STOMP_quadrangle_with_initial_recurrence;
                this.type = QUADRANGLE_WITH_INITIAL_RECURRENCE;
            }
            else if (i + width <= 0 and i + height > 0)
            {
                // Case quadrangle without initial recurrence
                // |\ 
                // | \
                // |  \
                //  \  \
                //   ---
                this.STOMP = this.STOMP_quadrangle_without_initial_recurrence;
                this.type = QUADRANGLE_WITHOUT_INITIAL_RECURRENCE;
            }
            else if (i + width > 0 and i + height > 0)
            {
                // Case polygon
                // -----
                // |    \
                //  \    \
                //   -----
                this.STOMP = this.STOMP_polygon;
                this.type = POLYGON;
            }
        }
        else if (i + width + height > n)
        {
            // Case parallelogram truncated on the right
            this.STOMP = this.STOMP_right_truncated_parallelogram;
            this.type = RIGHT_TRUNCATED_PARALLELOGRAM;
        }
        else
        {
            // Case parallelogram
            this.STOMP = this.STOMP_parallelogram;
            this.type = PARALLELOGRAM;
        }
    }
    /**
     * @brief Destructor
     */
    ~block()
    {
        // Destructor
    }

    /**
     * @brief Get the row needed for the recurrence in the block
     */
    std::vector<T> get_row()
    {
        return this.row;
    }

    /**
     * @brief Get the array of minimum per row in the block
     */
    std::vector<T> get_local_min_rows()
    {
        return this.local_min_row;
    }

    /**
     * @brief Get the type of the block
     */
    int get_type()
    {
        return this.type;
    }

private:
    const int type;              // type of the block
    const int global_i;                     // i coordinate top left corner
    const int global_j;                     // j coordinate top left corner
    const int width;                        // width
    const int height;                       // height
    const int n;                            // length of the distance matrix (the number of sequences )
    const int m;                            // the window size
    const std::vector<T> first_row;         // the first row of the distance matrix
    const std::vector<T> initial_row;       // the initial row for the recurrence
    const std::vector<T> time_series;       // the time series
    std::vector<T> row;                     // the row for the recurrence in the block
    std::vector<min_pair<T>> local_min_row; // the array of minimum per row in the block

    /**
     * @brief Update the minimum value and index of the pair
     * @param index the index of the minimum value to compare
     * @param min the pair to update
     */
    void update_min(int index, min_pair<T> &min)
    {
        if (this.row[index] < min.value)
        {
            min.value = this.row[index];
            min.index = index;
        }
    }

    /**
     * @brief Update the row for the recurrence in the block
     * @param j the index of the element to update in the row
     * @param global_i the global i coordinate
     * @param global_j the global j coordinate
     */
    void update_row(int j, int global_i, int global_j)
    {
        // Compute the elements to remove (prev) and the elements to add (next)
        auto prev_data = this.time_series[global_j - 1] - this.time_series[global_i - 1];
        auto next_data = this.time_series[global_j + this.m - 1] - this.time_series[global_i + this.m - 1];
        // Update the row following the recurrence
        this.row[j] += (next_data * next_data - prev_data * prev_data);
    }

    /**
     * @brief Update the recurrence of the row and compute the minimum of the row
     * @param start the start index of the row
     * @param end the end index of the row
     * @param i the index of the row
     * @param min the pair to update
     */
    void compute_row(int start, int end, int i, min_pair<T> &min)
    {
        for (int j = start; j < end; ++j)
        {
            update_row(j, this.global_i + i, this.global_j + j);
            update_min(j, min);
        }
        this.local_min_row[i] = min;
    }

    /**
     * @brief Initialize the block with the first row of the distance matrix and compute the minimum
     * @param start the start index of the row
     * @param end the end index of the row
     * @param min the pair to update
     */
    void initialize_with_first_row(int start, int end, min_pair<T> &min)
    {
        for (int j = start; j < end; ++j)
        {
            this.row[j] = this.first_row[this.global_j + j];
            update_min(j, min);
        }
        this.local_min_row[0] = min;
    }

    /**
     * @brief Initialize the block with the initial row for the recurrence and compute the minimum
     * @param start the start index of the row
     * @param end the end index of the row
     * @param min the pair to update
     */
    void initialize_with_initial_row(int start, int end, min_pair<T> &min)
    {
        int global_j;
        for (int j = start; j < end; ++j)
        {
            global_j = this.global_j + j;
            // Compute the elements to remove (prev) and the elements to add (next)
            auto prev_data = this.time_series[global_j - 1] - this.time_series[global_i - 1];
            auto next_data = this.time_series[global_j + this.m - 1] - this.time_series[global_i + this.m - 1];
            // Update the row following the recurrence
            this.row[j] = this.initial_row + (next_data * next_data - prev_data * prev_data);
            update_min(j, min);
        }
        this.local_min_row[0] = min;
    }
    /**
     * @brief Compute the minimum per row in the block in the parallelogram case
     */
    void STOMP_parallelogram()
    {
        min_pair<T> min = {-1, std::numeric_limits<T>::max()};
        // Initialize the first row
        if (this.global_i == 0)
        {
            // Case first row of the distance matrix
            initialize_with_first_row(0, this.width, min);
        }
        else
        {
            // Case initialize with the given initial row
            initialize_with_initial_row(0, this.width, min);
        }
        for (int i = 1; i < this.height; ++i)
        {
            min_pair<T> min = {-1, std::numeric_limits<T>::max()};
            compute_row(0, this.width, i, min);
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the triangle case
     */
    void STOMP_triangle()
    {
        int first_line = -(this.global_j + this.width);
        min_pair<T> min = {-1, std::numeric_limits<T>::max()};
        for (int i = 0; i < first_line; ++i)
        {
            this.local_min_row[i] = min;
        }
        int elem_per_row = 1;
        int max_j = this.witdh + this.j;
        int current_start;
        for (int i = first_line; i < this.height; ++i)
        {
            max_j += i;
            current_start = max_j - elem_per_row;
            // Compute the first element of the row after the truncation
            this.row[current_start] = this.first_row[this.global_i + i];
            update_min(current_start, min);

            // Compute the rest of the row
            compute_row(current_start + 1, max_j, i, min);
            elem_per_row++;
        }
    }
    /**
     * @brief Compute the minimum per row in the block in the polygon case
     */
    void STOMP_polygon()
    {
        int global_i, global_j;
        int start = -this.global_j;
        min_pair<T> min = {-1, std::numeric_limits<T>::max()};
        // intialize the first row
        initialize_with_initial_row(start, this.width, min);
        // Compute the first truncated rows
        // |-------
        // |       \
        // |        \
        // |---------|
        //
        for (int i = 1; i < start; ++i)
        {
            min_pair<T> min = {-1, std::numeric_limits<T>::max()};
            // Compute the first element of the row after the truncation
            this.row[start] = this.first_row[this.global_j + start];
            update_min(start, min);
            // Compute the rest of the row
            compute_row(start + 1, this.width, i, min);
        }
        // Compute the rest of the parallelogram
        for (int i = start; i < this.height; ++i)
        {
            min_pair<T> min = {-1, std::numeric_limits<T>::max()};
            compute_row(0, this.width, i, min);
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the quadrangle case without initial recurrence
     */
    void STOMP_quadrangle_without_initial_recurrence()
    {
        int first_line = -(this.global_j + this.width);
        // Part of the parallelogram on the left of the trucation
        for (int i = 0; i < first_line; ++i)
        {
            min_pair<T> min = {-1, std::numeric_limits<T>::max()};
            this.local_min_row[i] = min;
        }
        // Triangle part
        int elem_per_row = 1;
        for (int i = first_line; i < first_line + this.width; ++i)
        {
            min_pair<T> min = {-1, std::numeric_limits<T>::max()};
            int max_j = this.witdh + this.j + i;
            // First element of the row after the truncation
            this.row[max_j - elem_per_row] = this.first_row[this.global_i + i];
            update_min(max_j - elem_per_row, min);
            // Compute the rest of the row
            compute_row(max_j - elem_per_row + 1, max_j, i, min);
            elem_per_row++;
        }
        // Parallelogram part
        for (int i = first_line + this.width; i < this.height; ++i)
        {
            min_pair<T> min = {-1, std::numeric_limits<T>::max()};
            compute_row(0, this->width, i, min);
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the quadrangle case with initial recurrence
     */
    void STOMP_quadrangle_with_initial_recurrence()
    {
        int elem_per_row = this.width + this.global_j;
        int max_j = this.witdh;
        min_pair<T> min = {-1, std::numeric_limits<T>::max()};

        // Initialize the first row
        this.row[max_j - elem_per_row] = this.first_row[this.global_i];
        update_min(max_j - elem_per_row, min);
        for (int j = max_j - elem_per_row + 1; j < max_j; ++j)
        {
            this.row[j] = this.initial_row[j];
            update_min(j, min);
        }
        this.local_min_row[0] = min;

        // Compute the rest of the quadrangle
        for (int i = 1; i < this.height; ++i)
        {
            min_pair<T> min = {-1, std::numeric_limits<T>::max()};
            // Compute the first element of the row
            this.row[max_j - elem_per_row] = this.first_row[this.global_i + i];
            update_min(max_j - elem_per_row, min);
            // Compute the rest of the row
            compute_row(max_j - elem_per_row + 1, max_j, i, min);
            elem_per_row++;
        }
    }

    /**
     * @brief Compute the minimum per row in the block in the right truncated parallelogram case
     */
    void STOMP_right_truncated_parallelogram()
    {
        int i_max = this.n - this.global_j;
        int j_max = this.n - this.global_j;
        min_pair<T> min = {-1, std::numeric_limits<T>::max()};
        if (this.global_i == 0)
        {
            initialize_with_first_row(0, this.width, min);
        }
        else
        {
            initialize_with_initial_row(0, j_max, min);
        }
        for (int i = 1; i < this.height; ++i)
        {
            min_pair<T> min = {-1, std::numeric_limits<T>::max()};
            j_max = this.n - (this.global_j + i);
            compute_row(0, j_max, 0, min);
        }
    }
};