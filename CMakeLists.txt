project(MatrixProfile CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

cmake_minimum_required(VERSION 3.19 )

option(${CMAKE_PROJECT_NAME}_BUILD_TESTS "Build tests" ON)
option(${CMAKE_PROJECT_NAME}_BUILD_EXPERIMENTS "Build experiments" ON)
option(${CMAKE_PROJECT_NAME}_BUILD_PYTHON "Build python bindings" ON)

 
find_package(OpenMP REQUIRED)
# Find the pybind11 package
if (${CMAKE_PROJECT_NAME}_BUILD_PYTHON)
    find_package(Python COMPONENTS Development NumPy REQUIRED)
    find_package(pybind11 REQUIRED)
endif()

# Compiler flags
list(APPEND CMAKE_CXX_FLAGS "-march=native") # -Wall -Wextra -Wpedantic")


# Add the src directory
add_subdirectory(src)
# Build the tests
if(${CMAKE_PROJECT_NAME}_BUILD_TESTS)
    add_subdirectory(tests)
endif()
# Build the experiments
if (${CMAKE_PROJECT_NAME}_BUILD_EXPERIMENTS)
    add_subdirectory(experiments)
endif()