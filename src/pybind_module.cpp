#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "matrix_profile.hpp"
#include "interval_matrix_profile.hpp"
#include <bimp.hpp>
#include <bimb_knn.hpp>
#include <span>

namespace py = pybind11;

template <typename T>
inline std::span<T> numpy_array_to_span(py::array_t<T> const &array)
{
    py::buffer_info buffer = array.request();
    // Ensure the array is contiguous
    if (buffer.ndim != 1)
    {
        throw std::runtime_error("Array must be 1-dimensional");
    }

    // Get pointer to the data and size
    T *data_ptr = static_cast<T *>(buffer.ptr);
    size_t size = static_cast<size_t>(buffer.size);

    // Create and return std::span
    return std::span<T>(data_ptr, size);
}

template <typename T>
inline auto output_cast(std::pair<std::vector<T>, std::vector<int>> result)
{
    std::vector<T> const &matrix_profile = std::get<0>(result);
    std::vector<int> const &profile_index = std::get<1>(result);

    py::array_t<T> py_matrix_profile = py::cast(matrix_profile);
    py::array_t<int> py_profile_index = py::cast(profile_index);
    py::tuple py_result = py::make_tuple(py_matrix_profile, py_profile_index);
    return py_result;
}

template <typename T>
inline auto matrix_profile_STOMP_wrapper(py::array_t<T> &time_series,
                                         const int window_size,
                                         const int exclude,
                                         const int block_width,
                                         const int block_height)
{

    std::span<T> time_series_span = numpy_array_to_span(time_series);
    auto result = matrix_profile_STOMP(time_series_span, window_size, block_width, block_height, exclude);

    return output_cast(result);
}

template <typename T>
auto interval_matrix_profile_STOMP_wrapper(py::array_t<T> &time_series,
                                           const int window_size,
                                           py::array_t<int> const &period_starts,
                                           const int interval_length,
                                           const int exclude)
{
    std::span<T> time_series_span = numpy_array_to_span(time_series);
    std::span<int> period_starts_span = numpy_array_to_span(period_starts);

    auto result = BIMP(time_series_span, window_size, period_starts_span, interval_length, exclude);

    return output_cast(result);
}

template <typename T>
auto interval_matrix_profile_STOMP_knn_wrapper(py::array_t<T> &time_series,
                                               const int window_size,
                                               py::array_t<int> const &period_starts,
                                               const int interval_length,
                                               const int exclude,
                                               const int k,
                                               const bool exclude_diagonal)
{
    std::span<T> time_series_span = numpy_array_to_span(time_series);
    std::span<int> period_starts_span = numpy_array_to_span(period_starts);

    auto result = BIMP_kNN(time_series_span, window_size, period_starts_span, interval_length, exclude, k);

    return output_cast(result);
}

template <typename T>
auto BIMP_kNN_wrapper(py::array_t<T> &time_series,
                       const int window_size,
                       py::array_t<int> const &period_starts,
                       const int interval_length,
                       const int exclude,
                       const int k,
                       const bool exclude_diagonal)
{
    std::span<T> time_series_span = numpy_array_to_span(time_series);
    std::span<int> period_starts_span = numpy_array_to_span(period_starts);

    auto result = BIMP_kNN(time_series_span, window_size, period_starts_span, interval_length, exclude, k, exclude_diagonal);
    return output_cast(result);
}

template <typename T>
auto imp_bad_nn_wrapper(py::array_t<T> &time_series,
                        const int window_size,
                        py::array_t<int> const &period_starts,
                        const int interval_length,
                        const int exclude,
                        const int k)
{
    std::span<T> time_series_span = numpy_array_to_span(time_series);
    std::span<int> period_starts_span = numpy_array_to_span(period_starts);

    auto result = interval_matrix_profile_brute_force_bad_NN(time_series_span, window_size, period_starts_span, interval_length, exclude, k);
    return output_cast(result);
}

template <typename T>
auto imp_bf_wrapper(py::array_t<T> &time_series,
                    const int window_size,
                    py::array_t<int> const &period_starts,
                    const int interval_length,
                    const int exclude)
{
    std::span<T> time_series_span = numpy_array_to_span(time_series);
    std::span<int> period_starts_span = numpy_array_to_span(period_starts);

    auto result = interval_matrix_profile_brute_force(time_series_span, window_size, period_starts_span, interval_length, exclude);
    return output_cast(result);
}

template <typename T>
void bind_imp_bf(py::module &m, const std::string &name)
{
    m.def(name.c_str(), &imp_bf_wrapper<T>,
          "Compute the Interval Matrix Profile using the brute-force algorithm. The parameters are as follows:\n\
        time_series: numpy array, the time series data,\n\
        window_size: int, the size of the subsequences,\n\
        period_starts: numpy array, an array with the starting index of each period,\n\
        interval_length: int, the length of the interval (parameter L of the paper),\n\
        exclude: int, the size of the exclusion zone.\n\n\
    Returns a tuple of two numpy arrays: the Interval Matrix Profile and the Interval Matrix Profile Index.",

          py::arg("time_series"), py::arg("window_size"),
          py::arg("period_starts"), py::arg("interval_length"),
          py::arg("exclude"));
}

template <typename T>
void bind_imp_bad_nn(py::module &m, const std::string &name)
{
    m.def(name.c_str(), &imp_bad_nn_wrapper<T>,
          "Compute the Interval Matrix Profile using the brute-force algorithm with bad nearest neighbors. The parameters are as follows:\n\
        time_series: numpy array, the time series data,\n\
        window_size: int, the size of the subsequences,\n\
        period_starts: numpy array, an array with the starting index of each period,\n\
        interval_length: int, the length of the interval (parameter L of the paper),\n\
        exclude: int, the size of the exclusion zone,\n\
        k: int, the kNN parameter.\n\n\
    Returns a tuple of two numpy arrays: the Interval Matrix Profile and the Interval Matrix Profile Index.",

          py::arg("time_series"), py::arg("window_size"),
          py::arg("period_starts"), py::arg("interval_length"),
          py::arg("exclude"), py::arg("k"));
}

template <typename T>
void bind_BIMP_kNN(py::module &m, const std::string &name)
{
    m.def(name.c_str(), &BIMP_kNN_wrapper<T>,
          "Compute the kNN Interval Matrix Profile using the kNN BIMP algorithm. The parameters are as follows:\n\
        time_series: numpy array, the time series data,\n\
        window_size: int, the size of the subsequences,\n\
        period_starts: numpy array, an array with the starting index of each period,\n\
        interval_length: int, the lentgth of the interval (parameter L of the paper),\n\
        exclude: int, the size of the exclusion zone,\n\
        k: int, the kNN parameter.\n\
        exclude_diagonal, boolean, to exclude the diagonal blocks.\n\n\
    Returns a tuple of two numpy arrays: the Interval Matrix Profile and the Interval Matrix Profile Index.",

          py::arg("time_series"), py::arg("window_size"),
          py::arg("period_starts"), py::arg("interval_length"),
          py::arg("exclude"), py::arg("k"), py::arg("exclude_diagonal"));
}

template <typename T>
auto BIMP_kNN_left_wrapper(py::array_t<T> &time_series,
                           const int window_size,
                           py::array_t<int> const &period_starts,
                           const int interval_length,
                           const int exclude,
                           const int k,
                           const bool exclude_diagonal)
{
    std::span<T> time_series_span = numpy_array_to_span(time_series);
    std::span<int> period_starts_span = numpy_array_to_span(period_starts);

    auto result = left_BIMP_kNN(time_series_span, window_size, period_starts_span, interval_length, exclude, k, exclude_diagonal);
    return output_cast(result);
}

template <typename T>
auto BIMP_kNN_right_wrapper(py::array_t<T> &time_series,
                            const int window_size,
                            py::array_t<int> const &period_starts,
                            const int interval_length,
                            const int exclude,
                            const int k,
                            const bool exclude_diagonal)
{
    std::span<T> time_series_span = numpy_array_to_span(time_series);
    std::span<int> period_starts_span = numpy_array_to_span(period_starts);

    auto result = right_BIMP_kNN(time_series_span, window_size, period_starts_span, interval_length, exclude, k, exclude_diagonal);
    return output_cast(result);
}


template <typename T>
auto znormalized_imp_bf(py::array_t<T> &time_series,
                        const int window_size,
                        py::array_t<int> const &period_starts,
                        const int interval_length,
                        const int exclude)
{
    std::span<T> time_series_span = numpy_array_to_span(time_series);
    std::span<int> period_starts_span = numpy_array_to_span(period_starts);

    auto result = z_normalized_IMP_bf(time_series_span, window_size, period_starts_span, interval_length, exclude);
    return output_cast(result);
}

template <typename T>
void bind_znormalized_bf(py::module &m, const std::string &name)
{
    m.def(name.c_str(), &znormalized_imp_bf<T>,
          "Compute the left z-Normalized Interval Matrix Profile using the brute-force algorithm. The parameters are as follows:\n\
        time_series: numpy array, the time series data,\n\
        window_size: int, the size of the subsequences,\n\
        period_starts: numpy array, an array with the starting index of each period,\n\
        interval_length: int, the length of the interval (parameter L of the paper),\n\
        exclude: int, the size of the exclusion zone.\n\n\
    Returns a tuple of two numpy arrays: the Interval Matrix Profile and the Interval Matrix Profile Index.",

          py::arg("time_series"), py::arg("window_size"),
          py::arg("period_starts"), py::arg("interval_length"),
          py::arg("exclude"));
}

template <typename T>
void bind_left_BIMP_kNN(py::module &m, const std::string &name)
{
    m.def(name.c_str(), &BIMP_kNN_left_wrapper<T>,
          "Compute the left kNN Interval Matrix Profile using the kNN BIMP algorithm. The parameters are as follows:\n\
        time_series: numpy array, the time series data,\n\
        window_size: int, the size of the subsequences,\n\
        period_starts: numpy array, an array with the starting index of each period,\n\
        interval_length: int, the length of the interval (parameter L of the paper),\n\
        exclude: int, the size of the exclusion zone,\n\
        k: int, the kNN parameter.\n\
        exclude_diagonal, boolean, to exclude the diagonal blocks.\n\n\
    Returns a tuple of two numpy arrays: the Interval Matrix Profile and the Interval Matrix Profile Index.",

          py::arg("time_series"), py::arg("window_size"),
          py::arg("period_starts"), py::arg("interval_length"),
          py::arg("exclude"), py::arg("k"), py::arg("exclude_diagonal"));
}

template <typename T>
void bind_right_BIMP_kNN(py::module &m, const std::string &name)
{
    m.def(name.c_str(), &BIMP_kNN_right_wrapper<T>,
          "Compute the right kNN Interval Matrix Profile using the kNN BIMP algorithm. The parameters are as follows:\n\
        time_series: numpy array, the time series data,\n\
        window_size: int, the size of the subsequences,\n\
        period_starts: numpy array, an array with the starting index of each period,\n\
        interval_length: int, the length of the interval (parameter L of the paper),\n\
        exclude: int, the size of the exclusion zone,\n\
        k: int, the kNN parameter.\n\
        exclude_diagonal, boolean, to exclude the diagonal blocks.\n\n\
    Returns a tuple of two numpy arrays: the Interval Matrix Profile and the Interval Matrix Profile Index.",

          py::arg("time_series"), py::arg("window_size"),
          py::arg("period_starts"), py::arg("interval_length"),
          py::arg("exclude"), py::arg("k"), py::arg("exclude_diagonal"));
}

template <typename T>
void bind_matrix_profile_STOMP(py::module &m, const std::string &name)
{
    m.def(name.c_str(), matrix_profile_STOMP_wrapper<T>,
          "Compute the Matrix Profile using the STOMP algorithm. The parameters are as follows:\n\
        time_series: numpy array, the time series data,\n\
        window_size: int, the size of the subsequences,\n\
        exclude: int, the size of the exclusion zone,\n\
        block_width: int, the width of the blocks,\n\
        block_height: int, the height of the blocks.\n\n\
        Returns a tuple of two numpy arrays: the Matrix Profile and the Matrix Profile Index.",

          py::arg("time_series"), py::arg("window_size"),
          py::arg("exclude"), py::arg("block_width"),
          py::arg("block_height"));
}

template <typename T>
void bind_interval_matrix_profile_STOMP(py::module &m, const std::string &name)
{
    m.def(name.c_str(), &interval_matrix_profile_STOMP_wrapper<T>,
          "Compute the Interval Matrix Profile using the STOMP algorithm. The parameters are as follows:\n\
        time_series: numpy array, the time series data,\n\
        window_size: int, the size of the subsequences,\n\
        period_starts: numpy array, an array with the starting index of each period,\n\
        interval_length: int, the lentgth of the interval (parameter L of the paper),\n\
        exclude: int, the size of the exclusion zone.\n\n\
    Returns a tuple of two numpy arrays: the Interval Matrix Profile and the Interval Matrix Profile Index.",

          py::arg("time_series"), py::arg("window_size"),
          py::arg("period_starts"), py::arg("interval_length"),
          py::arg("exclude"));
}

template <typename T>
void bind_interval_matrix_profile_STOMP_knn(py::module &m, const std::string &name)
{
    m.def(name.c_str(), &interval_matrix_profile_STOMP_knn_wrapper<T>,
          "Compute the Interval Matrix Profile using the STOMP algorithm. The parameters are as follows:\n\
        time_series: numpy array, the time series data,\n\
        window_size: int, the size of the subsequences,\n\
        period_starts: numpy array, an array with the starting index of each period,\n\
        interval_length: int, the lentgth of the interval (parameter L of the paper),\n\
        exclude: int, the size of the exclusion zone.\n\
        k: int, the kNN parameter.\n\
        exclude_diagonal, boolean, to exclude the diagonal blocks.\n\n\
    Returns a tuple of two numpy arrays: the Interval Matrix Profile and the Interval Matrix Profile Index.",

          py::arg("time_series"), py::arg("window_size"),
          py::arg("period_starts"), py::arg("interval_length"),
          py::arg("exclude"), py::arg("k"), py::arg("exclude_diagonal"));
}

PYBIND11_MODULE(libimp, m)
{
    m.doc() = "Pybind11 library for the Interval Matrix Profile (IMP). There is currently only the STOMP implementation for float and double data. It is parallel";

    bind_matrix_profile_STOMP<float>(m, "STOMP_float");
    bind_matrix_profile_STOMP<double>(m, "STOMP_double");
    bind_interval_matrix_profile_STOMP<float>(m, "imp_STOMP_float");
    bind_interval_matrix_profile_STOMP<double>(m, "imp_STOMP_double");

    bind_BIMP_kNN<float>(m, "BIMP_kNN_float");
    bind_BIMP_kNN<double>(m, "BIMP_kNN_double");

    bind_left_BIMP_kNN<float>(m, "left_BIMP_kNN_float");
    bind_left_BIMP_kNN<double>(m, "left_BIMP_kNN_double");

    bind_right_BIMP_kNN<float>(m, "right_BIMP_kNN_float");
    bind_right_BIMP_kNN<double>(m, "right_BIMP_kNN_double");

    bind_znormalized_bf<float>(m, "znormalized_bf_float");
    bind_znormalized_bf<double>(m, "znormalized_bf_double");

    bind_imp_bad_nn<double>(m, "imp_bad_nn_double");
    
    bind_imp_bf<double>(m, "imp_bf_double");
}