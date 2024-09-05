project(MatrixProfile CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

cmake_minimum_required(VERSION 3.19 )

 
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

list(APPEND CMAKE_CXX_FLAGS "-march=native") # -Wall -Wextra -Wpedantic")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(experiments)


