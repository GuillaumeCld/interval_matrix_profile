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
    test_block
    test_block.cpp
)
target_link_libraries(
    test_block
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_block)
