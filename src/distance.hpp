#pragma once

#include <iostream>
#include <span>
#include <cmath>   // sqrt
#include <numeric> // inner_product
#include <cassert>
/**
 * @brief Compute the Euclidean distance between two vectors x, y.
 *
 * @tparam T
 * @param x first vector
 * @param y second vector
 * @return the scalar distance
 */
template <typename T, std::size_t N>
inline auto euclideanDistance(std::span<T, N> const &x, std::span<T, N> const &y) -> T
{

    static_assert(std::is_floating_point<T>::value, " works only with floating values");
    // T distance = std::inner_product(x.begin(), x.end(), y.begin(), T(0));
    auto distance = T(0);
    for (int i = 0; i < x.size(); ++i)
    {
        distance += std::pow((x[i] - y[i]), 2);
    }
    return std::sqrt(distance);
}

/**
 * @brief Compute the Euclidean distance between two vectors x, y.
 *
 * @tparam T
 * @param x
 * @param y
 * @return double
 */
template <typename T, std::size_t N>
inline auto dotProduct(std::span<T, N> const &x, std::span<T, N> const &y) -> T
{
    static_assert(std::is_floating_point<T>::value, " works only with floating values");

    auto distance{T(0.)};
    for (int i = 0; i < x.size(); ++i)
    {
        distance += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return distance;
}

/**
 * @brief Compute the z-normalized Euclidean distance between two vectors x, y.
 *
 * @tparam T
 * @param x
 * @param y
 * @return double
 */
template <typename T, std::size_t N>
inline auto znormalized_euclideanDistance(std::span<T, N> const &x, std::span<T, N> const &y) -> T
{
    static_assert(std::is_floating_point<T>::value, " works only with floating values");

    auto mean_x = std::accumulate(x.begin(), x.end(), T(0)) / x.size();
    auto mean_y = std::accumulate(y.begin(), y.end(), T(0)) / y.size();

    auto std_x = T(0);
    auto std_y = T(0);

    for (int i = 0; i < x.size(); ++i)
    {
        std_x += (x[i] - mean_x) * (x[i] - mean_x);
        std_y += (y[i] - mean_y) * (y[i] - mean_y);
    }

    std_x = std::sqrt(std_x);
    std_y = std::sqrt(std_y);

    auto distance{T(0.)};
    for (int i = 0; i < x.size(); ++i)
    {
        auto tmp = (x[i] - mean_x) / std_x - (y[i] - mean_y) / std_y;
        distance += tmp * tmp;
    }
    return std::sqrt(distance);
}


/**
 * @brief Compute the z-normalized dot product between two vectors x, y.
 *
 * @tparam T
 * @param x
 * @param y
 * @return double
 */
template <typename T, std::size_t N>
inline auto znormalized_dotProduct(std::span<T, N> const &x, std::span<T, N> const &y) -> T
{
    static_assert(std::is_floating_point<T>::value, " works only with floating values");


    auto mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    auto mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

    auto std_x = 0.0;
    auto std_y = 0.0;

    for (int i = 0; i < x.size(); ++i)
    {
        std_x += (x[i] - mean_x) * (x[i] - mean_x);
        std_y += (y[i] - mean_y) * (y[i] - mean_y);
    }

    std_x = std::sqrt(std_x);
    std_y = std::sqrt(std_y);

    auto distance = 0.0;
    for (int i = 0; i < x.size(); ++i)
    {
        auto tmp = (x[i] - mean_x) / std_x - (y[i] - mean_y) / std_y;
        distance += tmp*tmp;
    }
    return distance;
}