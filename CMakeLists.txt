cmake_minimum_required(VERSION 2.8)
include(ExternalProject)
project(POD)

# add Google Test as external library
ExternalProject_Add(gtest_ext
    SVN_REPOSITORY http://googletest.googlecode.com/svn/tags/release-1.7.0
    BINARY_DIR "${CMAKE_BINARY_DIR}/third-party/gtest-build"
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/gtest-src"
    INSTALL_COMMAND ""
    CMAKE_ARGS "${gtest_cmake_args}"
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    )
set(GTEST_INCLUDE_DIR
        "${CMAKE_BINARY_DIR}/third-party/gtest-src/include"
    )
set(GTEST_LIBRARY_DIR
        "${CMAKE_BINARY_DIR}/third-party/gtest-build"
    )

# add options to C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall")

# find boost
find_package(Boost COMPONENTS program_options iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# find mpi
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# include eigen
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/eigen")

# generate executable & link libraries
add_executable(pod src/pod.cpp)
target_link_libraries(pod ${Boost_LIBRARIES})
target_link_libraries(pod ${MPI_CXX_LIBRARIES})

# generate tests
enable_testing()
include_directories(${GTEST_INCLUDE_DIR})
add_executable(pod_test test/pod_test.cpp)
target_link_libraries(pod_test ${Boost_LIBRARIES})
target_link_libraries(pod_test ${MPI_CXX_LIBRARIES})
target_link_libraries(pod_test
    ${Boost_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${GTEST_LIBRARY_DIR}/libgtest.a
    ${GTEST_LIBRARY_DIR}/libgtest_main.a
    pthread
    )
add_test(test1 pod_test)
