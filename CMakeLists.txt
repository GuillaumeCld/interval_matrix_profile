project(MatrixProfile)
set(CMAKE_CXX_STANDARD 20)
cmake_minimum_required(VERSION 3.22)
find_package(OpenMP REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)

