
include_directories(${CMAKE_SOURCE_DIR}/src/)

add_executable(
    experiments
    computation_comparison.cpp
)
target_link_libraries(
    experiments
    OpenMP::OpenMP_CXX
)
