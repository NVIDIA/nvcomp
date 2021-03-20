set(GPU_ARCHS "60")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "8")
  set(GPU_ARCHS ${GPU_ARCHS} "70")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
  set(GPU_ARCHS ${GPU_ARCHS} "80")
endif()

file(GLOB CUDA_SOURCES
  ./*.cu
  ./lowlevel/*.cu)
file(GLOB CPP_SOURCES
  ./*.cpp
  ./lowlevel/*.cpp)

add_library(nvcomp SHARED ${CUDA_SOURCES} ${CPP_SOURCES})
set_property(TARGET nvcomp PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
target_compile_options(nvcomp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -Xcompiler -pthread>)
target_include_directories(nvcomp PUBLIC "../include")
target_include_directories(nvcomp PUBLIC "./")
if(DEFINED CUB_DIR)
  target_include_directories(nvcomp PUBLIC "${CUB_DIR}")
endif()
target_include_directories(nvcomp PUBLIC "${CUDAToolkit_INCLUDE_DIRS}")

if (ENABLE_BITCOMP)
  target_link_libraries(nvcomp PUBLIC bitcomp)
endif()

add_subdirectory("test")
