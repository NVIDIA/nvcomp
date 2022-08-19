file(GLOB GDEFLATE_CPU_SOURCES *gdeflate_cpu*.cpp *gdeflate_cpu*.cu)

# Add gdeflate CPU executables 
# gdeflate CPU examples require gdeflate and zlib libraries
# skip gdeflate CPU examples if required libraries are missing
# Also, nvcomp_gdeflate_cpu is not supported by Windows nvcomp
if (ZLIB_FOUND AND NOT MSVC)
  foreach(EXAMPLE_SOURCE ${GDEFLATE_CPU_SOURCES})
    # cut off suffixes
    string(REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
    string(REPLACE ".cu" "" EXAMPLE_NAME ${EXAMPLE_NAME})
    get_filename_component(BARE_NAME ${EXAMPLE_NAME} NAME)
    add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp::nvcomp CUDA::cudart)
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp::nvcomp_gdeflate_cpu ZLIB::ZLIB)
  endforeach(EXAMPLE_SOURCE ${GDEFLATE_CPU_SOURCES})
elseif(NOT MSVC)
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
# LZ4 CPU code is not supported by Windows nvcomp
if (LZ4_INCLUDE_DIR AND LZ4_LIBRARY AND NOT MSVC) 
  # lz4 CPU example requires lz4 libraries
  add_executable(lz4_cpu_compression lz4_cpu_compression.cu)
  target_link_libraries(lz4_cpu_compression PRIVATE nvcomp::nvcomp CUDA::cudart)
  target_include_directories(lz4_cpu_compression PRIVATE ${LZ4_INCLUDE_DIR})
  target_link_libraries(lz4_cpu_compression PRIVATE ${LZ4_LIBRARY})
  add_executable(lz4_cpu_decompression lz4_cpu_decompression.cu)
  target_link_libraries(lz4_cpu_decompression PRIVATE nvcomp::nvcomp CUDA::cudart)
  target_include_directories(lz4_cpu_decompression PRIVATE ${LZ4_INCLUDE_DIR})
  target_link_libraries(lz4_cpu_decompression PRIVATE ${LZ4_LIBRARY})
elseif(NOT MSVC)
  message(WARNING "Skipping building LZ4 CPU example, as no LZ4 library was found.")
endif()

# Add quickstart example
add_executable(low_level_quickstart_example low_level_quickstart_example.cpp)
target_link_libraries(low_level_quickstart_example PRIVATE nvcomp::nvcomp CUDA::cudart)

add_executable(high_level_quickstart_example high_level_quickstart_example.cpp)
target_link_libraries(high_level_quickstart_example PRIVATE nvcomp::nvcomp CUDA::cudart)
