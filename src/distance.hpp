#pragma once

#include <iostream>
#include <span>
#include <cmath> // sqrt
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
template<typename T, std::size_t N> 
inline auto euclideanDistance(const std::span<T, N> &x, const std::span<T, N>& y) -> T{

    static_assert(std::is_floating_point<T>::value, " works only with floating values");
    // T distance = std::inner_product(x.begin(), x.end(), y.begin(), T(0));
    auto distance = T(0);
    for (int i = 0; i < x.size(); ++i) {
        distance += std::pow((x[i] - y[i]), 2) ;
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
template<typename T, std::size_t N> 
inline auto dotProduct(const std::span<T, N> &x, const std::span<T, N>& y) -> T{
    static_assert(std::is_floating_point<T>::value, " works only with floating values");

    auto distance = T(0);
    for (int i = 0; i < x.size(); ++i) {
        distance += std::pow((x[i] - y[i]), 2) ;
    }

    return distance;
}