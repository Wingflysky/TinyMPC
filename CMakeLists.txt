cmake_minimum_required(VERSION 3.0.0)
list(APPEND CMAKE_MESSAGE_CONTEXT slap)
project(TinyMPC VERSION 0.2.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(include/Eigen)

add_subdirectory(src/tinympc)
add_subdirectory(examples)
add_subdirectory(test)