set(GPU_ARCHS "60")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "8")
  set(GPU_ARCHS ${GPU_ARCHS} "70")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
  set(GPU_ARCHS ${GPU_ARCHS} "80")
endif()

file(GLOB CUDA_SOURCES *.cu)
file(GLOB CPP_SOURCES *.cpp)

add_library(nvcomp SHARED ${CUDA_SOURCES} ${CPP_SOURCES})
set_property(TARGET nvcomp PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
target_compile_options(nvcomp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -Xcompiler -pthread>)
target_include_directories(nvcomp PUBLIC "../include")
if(DEFINED CUB_DIR)
  target_include_directories(nvcomp PUBLIC "${CUB_DIR}")
endif()
target_include_directories(nvcomp PUBLIC "${CUDAToolkit_INCLUDE_DIRS}")

add_subdirectory("test")
