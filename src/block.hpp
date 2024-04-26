template<typename T>
class block {
public:
    block(int n, int m, int exclude, int i, int j, int ID, int width, int height, T* first_row; T* initial_row, T* time_series)
        : n(n), m(m), global_i(i), global_j(j), ID(ID), width(width), height(height),
          local_min_row(new T[height]), local_min_col(new T[width]), row(new T[width]) {


        if (i < 0) {
            // Case parallelogram truncated on the left
            if (i+width <= 0 and i+height <= 0)  {
                // Case triangle
                this.STOMP = this.STOMP_triangle;
            } else if (i+width > 0 and i+height <=0) {
                // Case quadrangle with initial recurrence

            } else if (i+width <= 0 and i+height > 0) {
                // Case quadrangle without initial recurrence

            } else if (i+width > 0 and i+height > 0) {
                // Case polygon

            }
        } else if (i + width + height > n) {
            // Case parallelogram truncated on the right
            this.STOMP = this.STOMP_right_truncated_parallelogram;
        } else {
            // Case parallelogram
            this.STOMP = this.STOMP_parallelogram;
        }
    }

    ~block() {
        delete[] local_min_row;
        delete[] local_min_col;
    }

    private:
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

    void STOMP();

    void STOMP_parallelogram(){
        int start = ( this.i == 0 ) ? 1 : 0;
        for (int i = start, i < this.height; ++i) {
            int global_i = this.global_i + i;
            auto min = std::numeric_limits<T>::max();
            int argmin = -1;
            for (int j = 0; j < this.width; ++j) {
                int global_j = this.global_j + j;
                auto prev_data = this.time_series[global_j-1] - this.time_series[global_j-1];
                auto next_data = this.time_series[global_j+this.m-1] - this.time_series[global_i+this.m-1];
                
                this.row[j] = this.row[j] - prev_data*prev_data + next_data*next_data;
                if (this.row[j] < min) {
                    min = row[j];
                    argmin = j;
                }
            }
            this.local_min_row[i] = min;
        }
    }
    void STOMP_right_truncated_parallelogram(){
    }

    void STOMP_triangle(){
        int first_line = -(this.global_j + this.width);

        for (int i = 0; i < first_line; ++i) {
            local_min_row[i] = std::numeric_limits<T>::max();
        }
        int elem_per_row = 1;
        for (int i = first_line; i < this.height; ++i) {
            int global_i = this.global_i + i;

            int argmin = -1;
       
            int max_j = this.witdh + this.i+i; 

            this.row[max_j-elem_per_row] = this.first_row[this.global_i + i];
            auto min = this.row[max_j-elem_per_row];

            for (int j = max_j-elem_per_row+1; j < max_j; ++j) {
                int global_j = this.global_j + j;
                auto prev_data = this.time_series[global_j-1] - this.time_series[global_j-1];
                auto next_data = this.time_series[global_j+this.m-1] - this.time_series[global_i+this.m-1];

                this.row[j] = this.row[j] - prev_data*prev_data + next_data*next_data;
                if (this.row[j] < min) {
                    min = row[j];
                    argmin = j;
                }
            }
            elem_per_row++;
            this.local_min_row[i] = min;
        }
    }

};