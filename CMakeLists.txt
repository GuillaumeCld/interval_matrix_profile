project(MatrixProfile CXX)
set(CMAKE_CXX_STANDARD 20)
cmake_minimum_required(VERSION 3.22)
find_package(OpenMP REQUIRED)
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_CXX_FLAGS "-march=native") # -Wall -Wextra -Wpedantic")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(experiments)


