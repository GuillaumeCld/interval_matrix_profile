include_directories(
        ${PROJECT_SOURCE_DIR}/src
)

add_executable(main main.cpp distance.hpp matrix_profile.hpp)

target_compile_options(main PRIVATE -O3 -march=native ${OpenMP_CXX_FLAGS})
target_link_libraries(main PRIVATE ${OpenMP_CXX_FLAGS})


# -march=native