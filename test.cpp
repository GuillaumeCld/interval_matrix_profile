#include <iostream>
#include <vector>
#include <limits>
#include <span>
// Define T as double for simplicity
using T = double;

// Define a min_pair struct to store the minimum value and its index
template <typename T>
struct min_pair
{
    T value;
    int index;
    min_pair() : value(std::numeric_limits<T>::max()), index(-1) {}
};

T dotProduct(std::vector<T> x, std::vector<T> y)
{

    auto distance{T(0.)};
    for (int i = 0; i < x.size(); ++i)
    {
        distance += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return distance;
}

// Update a single element in the row based on the recurrence
void update_row_element(int j, int global_i, int global_j, int m, std::vector<T> &time_series, std::vector<T> &row)
{
    T prev_data = time_series[global_i - 1] - time_series[global_j - 1];
    T next_data = time_series[global_i + m - 1] - time_series[global_j + m - 1];
    row[j] += (next_data * next_data - prev_data * prev_data);
}

// Update the minimum value and index in min if a smaller value is found
void update_min(int j, min_pair<T> &min, int global_i, int global_j, std::vector<T> &row)
{
    if (row[j] < min.value)
    {
        min.value = row[j];
        min.index = j;
    }
}

// Compute the recurrence in the row and find the minimum
void compute_row(int start, int end, int i, min_pair<T> &min, int global_i_base, int global_j_base, int m, std::vector<T> &time_series, std::vector<T> &row)
{
    int global_i = global_i_base + i;
    int global_j = global_j_base + start + i;
    for (int j = start; j < end; ++j)
    {
        T prev_data = time_series[global_i - 1] - time_series[global_j - 1];
        T next_data = time_series[global_i + m - 1] - time_series[global_j + m - 1];
        row[j] += (next_data * next_data - prev_data * prev_data);
        ++global_j;
    }
    global_j = global_j_base + start + i;
    for (int j = start; j < end; ++j)
    {
        update_min(j, min, global_i, global_j, row);
        ++global_j;
    }
}

auto modified_AAMP(std::vector<T> &time_series,
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

    std::vector<T> row_values(n_sequence, 0);

    for (int i_period = 0; i_period < n_periods_i; ++i_period)
    {
        const int i_start = period_starts[i_period];
        const int i_end = (i_period == n_periods_i - 1) ? n_sequence : period_starts[i_period + 1];
        for (int i = i_start; i < i_end; ++i)
        {
            auto min = std::numeric_limits<T>::max();
            int min_index = 0;
            const int i_pos = i - i_start;
            for (int j = n_sequence - 1; j >= 0; --j)
            {
                // Compute Euclidean distance for the current sequence
                if (i == 0 or j == 0)
                {
                    // auto view = std::span(&time_series[i], window_size);
                    // const auto distance = dotProduct(view, std::span(&time_series[j], window_size));
                    row_values[j] = 0;
                }
                else
                {
                    const auto prev_data{time_series[i - 1] - time_series[j - 1]};
                    const auto next_data{time_series[i + window_size - 1] - time_series[j + window_size - 1]};
                    const auto distance = row_values[j - 1] + (next_data * next_data - prev_data * prev_data);
                    row_values[j] = distance;
                }
            }
            for (int j_period = 0; j_period < n_periods_i; ++j_period)
            {
                const int j_start = std::max(period_starts[j_period] + i_pos - half_interval, 0);
                const int j_end = std::min(period_starts[j_period] + i_pos + half_interval + 1, n_sequence);
                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const auto distance = row_values[j];
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
            }

            if (i_end - i <= half_interval)
            {
                const int j_start = 0;
                const int j_end = half_interval - (i_end - i) + 1;
                for (int j = j_start; j < j_end; ++j)
                {
                    const auto distance = row_values[j];
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
            }
            if (is_periodic and i - i_start < half_interval)
            {

                const int j_start = n + i_pos - half_interval;
                const int j_end = n_sequence;

                for (int j = j_start; j < j_end; ++j)
                {
                    // Compute Euclidean distance for the current sequence
                    const auto distance = row_values[j];
                    if (distance < min and (j < i - exclude or j > i + exclude))
                    {
                        min = distance;
                        min_index = j;
                    }
                }
            }

            matrix_profile[i] = min;
            profile_index[i] = min_index;
        }
    }

    return std::make_pair(matrix_profile, profile_index);
}

int main()
{
    int start = 0;
    // Sample time series and row
    std::vector<T> time_series;
    for (int i = 0; i < 1000; ++i)
    {
        time_series.push_back(i);
    }
    std::vector<T> row{0.0, 0.0, 0.0, 0.0, 0.0};
    int end = row.size();
    int i = 1; // Example row index
    int m = 2; // Example m value
    int global_i_base = 1;
    int global_j_base = 1;

    min_pair<T> min;
    compute_row(start, end, i, min, global_i_base, global_j_base, m, time_series, row);

    std::cout << "Minimum value in row: " << min.value << " at index " << min.index << std::endl;

    return 0;
}
