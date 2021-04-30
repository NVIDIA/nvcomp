set(GPU_ARCHS "60")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "8")
  set(GPU_ARCHS ${GPU_ARCHS} "70")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "9")
  set(GPU_ARCHS ${GPU_ARCHS} "75")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
  set(GPU_ARCHS ${GPU_ARCHS} "80")
endif()

file(GLOB CUDA_SOURCES
  ./*.cu
  ./lowlevel/*.cu
  ./highlevel/*.cu)
file(GLOB CPP_SOURCES
  ./*.cpp
  ./lowlevel/*.cpp
  ./highlevel/*.cpp)

if (MSVC_VERSION)
  # we have to export all symbols on windows for a dll
  add_library(nvcomp STATIC ${CUDA_SOURCES} ${CPP_SOURCES})
else()
  add_library(nvcomp SHARED ${CUDA_SOURCES} ${CPP_SOURCES})
endif()

set_property(TARGET nvcomp PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
target_compile_options(nvcomp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_include_directories(nvcomp PUBLIC "../include")
target_include_directories(nvcomp PUBLIC "./")
if(DEFINED CUB_DIR)
  target_include_directories(nvcomp PUBLIC "${CUB_DIR}")
endif()
target_include_directories(nvcomp PUBLIC "${CUDAToolkit_INCLUDE_DIRS}")

if (bitcomp_FOUND)
  target_link_libraries(nvcomp PUBLIC bitcomp)
endif()

if (GDEFLATE_FOUND)
  target_include_directories(nvcomp PUBLIC ${GDEFLATE_INCLUDE_DIRS})
  target_link_libraries(nvcomp PUBLIC ${GDEFLATE_LIBRARIES})
endif()

add_subdirectory("test")
add_subdirectory("highlevel/test")
add_subdirectory("lowlevel/test")
