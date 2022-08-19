cmake_minimum_required(VERSION 3.18)
project(nvcomp_examples LANGUAGES C CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_BENCHMARKS "Build benchmarks." OFF)

include(GNUInstallDirs)

list( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
)

find_package(CUDAToolkit REQUIRED)
find_package(nvcomp REQUIRED 2.3)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR
    "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()

# Using '-G' can cause the register count to overflow for some kernels
# and cause launch failures
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG};-G")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -t 0")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -t 0")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug build.")
else()
  message("Release build.")
  # this is required, as CMake sends CMAKE_CXX_FLAGS_DEBUG to nvcc
  # if no build type is set (but not to g++ for some reason).
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(NVCOMP_FIND_QUIET "QUIET")
if(DEFINED DEVEL AND NOT DEVEL EQUAL 0)
  unset(NVCOMP_FIND_QUIET )
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    # g++ warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    # this doesn't work with old style CMakeLists and cuda
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")

    # g++ warnings
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
    # this doesn't work with old style CMakeLists and cuda
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic")
  endif()

  
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror=all-warnings")
  endif()
  # TODO: add clang warnings
endif()

if (MSVC)
  # Use "/permissive-" in order to be closer to GCC/Clang in what the MSVC compiler accepts.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W3>" "$<$<COMPILE_LANGUAGE:CXX>:/permissive->" "$<$<COMPILE_LANGUAGE:CXX>:/D \"_CRT_SECURE_NO_WARNINGS\">")
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
if(NOT CUB_HEADER)
  message(FATAL_ERROR "Unable to locate 'cub/cub.cuh'. Please specify a path "
      "to cub via 'CUB_DIR'")
endif()

if(NOT DEFINED NVCOMP_EXTS_ROOT AND DEFINED ENV{NVCOMP_EXTS_ROOT})
  set(NVCOMP_EXTS_ROOT "$ENV{NVCOMP_EXTS_ROOT}")
endif()

find_package(ZLIB) # For Gdeflate CPU compression

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (UNIX)
  include(GNUInstallDirs)
endif()


add_subdirectory(examples)

if (DEFINED BUILD_BENCHMARKS) 
  add_subdirectory(benchmarks)
endif()
