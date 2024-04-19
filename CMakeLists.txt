project(MatrixProfile)
set(CMAKE_CXX_STANDARD 20)
cmake_minimum_required(VERSION 3.22)

find_package(OpenMP REQUIRED)

add_executable(main main.cpp distance.hpp matrix_profile.hpp)
target_include_directories(main PRIVATE /home/guillaume/Devel/MP_CPP/)

target_compile_options(main PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(main PRIVATE ${OpenMP_CXX_FLAGS})