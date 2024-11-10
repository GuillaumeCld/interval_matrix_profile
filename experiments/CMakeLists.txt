
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

add_executable(
    climate_pulse
    climate_pulse_imp.cpp
)

add_executable(
    numerical_error
    numerical_error.cpp
)

target_link_libraries(
    experiments
    OpenMP::OpenMP_CXX
)
target_link_libraries(
    climate_exp
    OpenMP::OpenMP_CXX
)
target_link_libraries(
    climate_pulse
    OpenMP::OpenMP_CXX
)
target_link_libraries(
    numerical_error
    OpenMP::OpenMP_CXX
)


add_executable(imp_time imp_time_comparison.cpp)
target_link_libraries(imp_time PRIVATE OpenMP::OpenMP_CXX)
