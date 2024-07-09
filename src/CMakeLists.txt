include_directories(
        ${PROJECT_SOURCE_DIR}/src
)

add_executable(main main.cpp distance.hpp matrix_profile.hpp)
add_executable(knn k_nn.cpp k_nn_lookup.hpp)

target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)