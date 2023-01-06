file(GLOB GDEFLATE_CPU_SOURCES *gdeflate_cpu*.cpp *gdeflate_cpu*.cu)

# Add gdeflate CPU executables 
# gdeflate CPU examples require gdeflate and zlib libraries
# skip gdeflate CPU examples if required libraries are missing
if (ZLIB_FOUND)
  foreach(EXAMPLE_SOURCE ${GDEFLATE_CPU_SOURCES})
    # cut off suffixes
    string(REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
    string(REPLACE ".cu" "" EXAMPLE_NAME ${EXAMPLE_NAME})
    get_filename_component(BARE_NAME ${EXAMPLE_NAME} NAME)
    add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp::nvcomp CUDA::cudart)
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp::nvcomp_gdeflate_cpu ZLIB::ZLIB)
  endforeach(EXAMPLE_SOURCE ${GDEFLATE_CPU_SOURCES})
else()
  message(WARNING "Skipping building gdeflate CPU example, as zlib library not found.")
endif()

option (BUILD_GDS_EXAMPLE "Build NVCOMP + GDS example" OFF)
if (BUILD_GDS_EXAMPLE)
  add_executable (nvcomp_gds nvcomp_gds.cu)
  target_link_libraries(nvcomp_gds PRIVATE nvcomp CUDA::cudart cufile)
endif()

# Add LZ4 CPU executable
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)
if (LZ4_INCLUDE_DIR AND LZ4_LIBRARY) 
  # lz4 CPU example requires lz4 libraries
  add_executable(lz4_cpu_compression lz4_cpu_compression.cu)
  target_link_libraries(lz4_cpu_compression PRIVATE nvcomp::nvcomp CUDA::cudart)
  target_include_directories(lz4_cpu_compression PRIVATE ${LZ4_INCLUDE_DIR})
  target_link_libraries(lz4_cpu_compression PRIVATE ${LZ4_LIBRARY})
  add_executable(lz4_cpu_decompression lz4_cpu_decompression.cu)
  target_link_libraries(lz4_cpu_decompression PRIVATE nvcomp::nvcomp CUDA::cudart)
  target_include_directories(lz4_cpu_decompression PRIVATE ${LZ4_INCLUDE_DIR})
  target_link_libraries(lz4_cpu_decompression PRIVATE ${LZ4_LIBRARY})
else()
  message(WARNING "Skipping building LZ4 CPU example, as no LZ4 library was found.")
endif()

# Add quickstart example
add_executable(low_level_quickstart_example low_level_quickstart_example.cpp)
target_link_libraries(low_level_quickstart_example PRIVATE nvcomp::nvcomp CUDA::cudart)

add_executable(high_level_quickstart_example high_level_quickstart_example.cpp)
target_link_libraries(high_level_quickstart_example PRIVATE nvcomp::nvcomp CUDA::cudart)
target_include_directories(high_level_quickstart_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(bitcomp_native_lossyFP bitcomp_native_lossyFP.cu)
target_link_libraries(bitcomp_native_lossyFP PRIVATE nvcomp::nvcomp)


# Add deflate example
find_path(LIBDEFLATE_INCLUDE_DIR NAMES libdeflate.h)
find_library(LIBDEFLATE_LIBRARY NAMES libdeflate)
if (ZLIB_FOUND AND LIBDEFLATE_INCLUDE_DIR AND LIBDEFLATE_INCLUDE_DIR)
  add_executable(deflate_cpu_compression deflate_cpu_compression.cu)
  if (NOT MSVC)
    target_link_libraries(deflate_cpu_compression PRIVATE nvcomp::nvcomp CUDA::cudart)
  else()
    target_link_libraries(deflate_cpu_compression PRIVATE nvcomp::nvcomp)
  endif()
  target_link_libraries(deflate_cpu_compression PRIVATE ZLIB::ZLIB)
  target_include_directories(deflate_cpu_compression PRIVATE ${LIBDEFLATE_INCLUDE_DIR})
  target_link_libraries(deflate_cpu_compression PRIVATE ${LIBDEFLATE_LIBRARY})
  target_include_directories(deflate_cpu_compression PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

  add_executable(deflate_cpu_decompression deflate_cpu_decompression.cu)
  if (NOT MSVC)
    target_link_libraries(deflate_cpu_decompression PRIVATE nvcomp::nvcomp CUDA::cudart)
  else()
    target_link_libraries(deflate_cpu_decompression PRIVATE nvcomp::nvcomp)
  endif()
  target_link_libraries(deflate_cpu_decompression PRIVATE ZLIB::ZLIB)
  target_include_directories(deflate_cpu_decompression PRIVATE ${LIBDEFLATE_INCLUDE_DIR})
  target_link_libraries(deflate_cpu_decompression PRIVATE ${LIBDEFLATE_LIBRARY})
  target_include_directories(deflate_cpu_decompression PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
