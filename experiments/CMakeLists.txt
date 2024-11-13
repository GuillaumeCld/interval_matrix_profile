
include_directories(${CMAKE_SOURCE_DIR}/src/)

add_executable(
    numerical_error
    numerical_error.cpp
)
target_link_libraries(
    numerical_error
    OpenMP::OpenMP_CXX
)

add_executable(
    imp_time 
    imp_time_comparison.cpp
)
target_link_libraries(
    imp_time 
    OpenMP::OpenMP_CXX
)
