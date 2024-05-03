include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


include_directories(${CMAKE_SOURCE_DIR}/src/)

enable_testing()

add_executable(
    tests
    test_block.cpp
    test_mp.cpp
)
target_link_libraries(
    tests
    GTest::gtest_main
    ${OpenMP_CXX_FLAGS}
)
target_compile_options(tests PRIVATE ${OpenMP_CXX_FLAGS})

include(GoogleTest)
gtest_discover_tests(tests)
