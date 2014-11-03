cmake_minimum_required(VERSION 2.8)
project(POD)

# add options to C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# find boost
find_package(Boost COMPONENTS program_options iostreams REQUIRED)
MESSAGE( STATUS "Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})
MESSAGE( STATUS "Boost_LIBRARIES: " ${Boost_LIBRARIES})
include_directories(${Boost_INCLUDE_DIRS})

# include eigen
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/eigen")

# generate executables
add_executable(pod src/pod.cpp)

# link libraries
target_link_libraries(pod ${Boost_LIBRARIES})
