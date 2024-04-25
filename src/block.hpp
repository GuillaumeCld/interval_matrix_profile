template<typename T>
class block {
public:
    block(int n, int m, int exclude, int i, int j, int ID, int width, int height, T* first_row; T* initial_row, T* time_series)
        : n(n), m(m), global_i(i), global_j(j), ID(ID), width(width), height(height),
          local_min_row(new T[height]), local_min_col(new T[width]), row(new T[width]) {
    }

    ~block() {
        delete[] local_min_row;
        delete[] local_min_col;
    }

public:
    int global_i;  // i coordinate
    int global_j;  // j coordinate
    int ID;  // ID
    int width;  // width
    int height;  // height
    int n;
    int m;
    T* first_row;
    T* initial_row;
    T* time_series;
    T* row;
    T* local_min_row;  // array of type T for local minimum row
    T* local_min_col;  // array of type T for local minimum column

void compute_minimums():

    // Iniate the first row for the induction
    int start = std::max(0, global_i);
    int end = std::min(global_i + width, n);
    int length = end - start;
    for (int j = length-1; j >= 0 ; --j) {
        row[j] = row[j-1] - std::pow(time_series[j-1]-time_series[i-1],2) + std::pow(time_series[j+m-1]-time_series[i+m-1],2);

    if (row[j] < local_min_row[i] and (global_j < global_i - exclude or global_j > global_i + exclude)) {
                local_min_row[i] = row[j];
            }

    // Induction step
    for (int i = 1; i < height; ++i) {
        local_min_row[i] = std::numeric_limits<T>::max();
        int start = std::max(0, global_i + i);
        int end = std::min(global_i + i + width, n);
        for (int j = start-end-1; j >= 0 ; --j) {
            row[j] = row[j-1] - std::pow(time_series[j-1]-time_series[i-1],2) + std::pow(time_series[j+m-1]-time_series[i+m-1],2);

        if (row[j] < local_min_row[i] and (global_j < global_i - exclude or global_j > global_i + exclude)) {
                    local_min_row[i] = row[j];
                }
        }

    }


};