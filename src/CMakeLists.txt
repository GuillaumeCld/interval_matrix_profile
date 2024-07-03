include_directories(
        ${PROJECT_SOURCE_DIR}/src
)

add_executable(main main.cpp distance.hpp matrix_profile.hpp)
add_executable(knn k_nn.cpp k_nn_lookup.hpp)

target_compile_options(main PRIVATE -march=native ${OpenMP_CXX_FLAGS})
target_link_libraries(main PRIVATE ${OpenMP_CXX_FLAGS})


# -march=native