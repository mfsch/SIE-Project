cmake_minimum_required(VERSION 2.8)
project(POD)

# add options to C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# find boost
find_package(Boost COMPONENTS program_options iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# find mpi
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# include eigen
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/eigen")

# generate executables
add_executable(pod src/pod.cpp)

# link libraries
target_link_libraries(pod ${Boost_LIBRARIES})
target_link_libraries(pod ${MPI_CXX_LIBRARIES})
