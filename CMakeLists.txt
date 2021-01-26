cmake_minimum_required(VERSION 3.18)
project(nvcomp LANGUAGES C CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
# Using '-G' can cause the register count to overflow for some kernels
# and cause launch failures
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG};-G")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug build.")
else()
  message("Release build.")
  # this is required, as CMake sends CMAKE_CXX_FLAGS_DEBUG to nvcc
  # if no build type is set (but not to g++ for some reason).
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(DEFINED DEVEL AND NOT DEVEL EQUAL 0)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    # g++ warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    # this doesn't work with old style CMakeLists and cuda
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
  endif()

  # TODO: add clang warnings
endif()


if(NOT DEFINED CUB_DIR AND DEFINED ENV{CUB_DIR})
  set(CUB_DIR ENV{CUB_DIR})
endif()

message(STATUS "Finding CUB")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11" AND NOT DEFINED CUB_DIR)
  message(FATAL_ERROR "When using a CUDA prior to CUDA 11, you must specify "
      "the path of cub via the 'CUB_DIR' variable (e.g., "
      "'cmake -DCUB_DIR=...' or 'export CUB_DIR=...'). You can download cub "
      "from 'https://github.com/thrust/cub'.")
endif()
find_path(CUB_HEADER NAMES "cub/cub.cuh" PATHS "${CUB_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
IF(NOT CUB_HEADER)
  message(FATAL_ERROR "Unable to locate 'cub/cub.cuh'. Please specify a path "
      "to cub via 'CUB_DIR'")
ENDIF()


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(COPY include/ DESTINATION ${CMAKE_BINARY_DIR}/include)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
