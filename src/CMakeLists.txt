set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
  -std=c++14
  --expt-extended-lambda
  -Xcompiler -pthread
  -gencode arch=compute_60,code=sm_60
)

IF(CUDA_VERSION_MAJOR GREATER 8)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=sm_70)
ENDIF(CUDA_VERSION_MAJOR GREATER 8)
IF(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_80,code=sm_80)
ENDIF(CUDA_VERSION_MAJOR GREATER_EQUAL 11)

file(GLOB CUDA_SOURCES *.cu)
file(GLOB CPP_SOURCES *.cpp)

cuda_add_library(nvcomp SHARED ${CUDA_SOURCES} ${CPP_SOURCES})
target_link_libraries(nvcomp)

add_subdirectory("test")
