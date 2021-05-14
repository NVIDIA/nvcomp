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

include_directories("./")

set_property(TARGET nvcomp PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
target_compile_options(nvcomp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_include_directories(nvcomp PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>")
if(DEFINED CUB_DIR)
  target_include_directories(nvcomp PRIVATE "${CUB_DIR}")
endif()
target_include_directories(nvcomp PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")

if (bitcomp_FOUND)
  target_include_directories(nvcomp INTERFACE ${BITCOMP_INCLUDE_DIRS})
  target_link_libraries(nvcomp PRIVATE bitcomp)
endif()

if (GDEFLATE_FOUND)
  target_include_directories(nvcomp INTERFACE ${GDEFLATE_INCLUDE_DIRS})
  target_link_libraries(nvcomp PRIVATE ${GDEFLATE_LIBRARIES})
endif()

if (BUILD_TESTS)
  add_subdirectory("test")
  add_subdirectory("highlevel/test")
  add_subdirectory("lowlevel/test")
endif()
