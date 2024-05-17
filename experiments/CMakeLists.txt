
include_directories(${CMAKE_SOURCE_DIR}/src/)

add_executable(
    experiments
    computation_comparison.cpp
)
target_link_libraries(
    experiments
    ${OpenMP_CXX_FLAGS}
)
target_compile_options(experiments PRIVATE ${OpenMP_CXX_FLAGS})
