cmake_minimum_required(VERSION 3.1)
project(nvcomp)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

include_directories(include)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

IF(CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug build.")
  # Using '-G' can cause the register count to overflow for some kernels
  # and cause launch failures
  # set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-G)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g)
ELSE()
  message("Release build.")
  # this is required, as CMake sends CMAKE_CXX_FLAGS_DEBUG to nvcc
  # if no build type is set (but not to g++ for some reason).
  set(CMAKE_BUILD_TYPE "Release")
ENDIF()

if (DEFINED DEVEL AND NOT DEVEL EQUAL 0)
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    # g++ warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    # this doesn't work with old style CMakeLists and cuda
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
  endif()

  # TODO: add clang warnings
endif()


IF (NOT DEFINED CUB_DIR AND DEFINED ENV{CUB_DIR})
  set(CUB_DIR ENV{CUB_DIR})
ENDIF()
if (DEFINED CUB_DIR)
  include_directories("${CUB_DIR}")
endif()

message(STATUS "Finding CUB")
IF(CUDA_VERSION_MAJOR LESS 11)
  IF (NOT DEFINED CUB_DIR)
    message(FATAL_ERROR "When using a CUDA prior to CUDA 11, you must specify "
       "the path of cub via the 'CUB_DIR' variable (e.g., "
       "'cmake -DCUB_DIR=...' or 'export CUB_DIR=...'). You can download cub "
       "from 'https://github.com/thrust/cub'.")
  ENDIF()
ENDIF()
find_path(CUB_HEADER NAMES "cub/cub.cuh" PATHS "${CUB_DIR}" "${CUDA_INCLUDE_DIRS}")
IF (NOT CUB_HEADER)
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
