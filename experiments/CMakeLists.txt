
include_directories(${CMAKE_SOURCE_DIR}/src/)

add_executable(
    experiments
    computation_comparison.cpp
)

add_executable(
    k_nn
    k_nn_time.cpp
)
add_executable(
    climate_exp
    climate_comparison.cpp
)

target_link_libraries(
    experiments
    OpenMP::OpenMP_CXX
)
target_link_libraries(
    climate_exp
    OpenMP::OpenMP_CXX
)
