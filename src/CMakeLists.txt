include_directories(
        ${PROJECT_SOURCE_DIR}/src
)

add_executable(main main.cpp distance.hpp matrix_profile.hpp)
add_executable(knn k_nn.cpp k_nn_lookup.hpp)


target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)

# Build the python bindings
if (${CMAKE_PROJECT_NAME}_BUILD_PYTHON)
    # Define the pybind11 module
    pybind11_add_module(libimp MODULE
        pybind_module.cpp  
        interval_matrix_profile.hpp  
    )
    # Link the pybind11 module with OpenMP
    target_link_libraries(libimp PRIVATE OpenMP::OpenMP_CXX)
endif()


