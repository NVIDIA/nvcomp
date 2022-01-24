/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "highlevel/SnappyHlifKernels.h"
#include "nvcomp_common_deps/hlif_shared.cuh"
#include "SnappyKernels.cuh"
#include "CudaUtils.h"

namespace nvcomp {

struct snappy_compress_wrapper : hlif_compress_wrapper {

private:
  nvcompStatus_t* status;

public:
  __device__ snappy_compress_wrapper(uint8_t* /*tmp_buffer*/, uint8_t* /*share_buffer*/, nvcompStatus_t* status)
   : status(status)
  {}
      
  __device__ void compress_chunk(
      uint8_t* tmp_output_buffer,
      const uint8_t* this_decomp_buffer,
      const size_t decomp_size,
      const size_t max_comp_chunk_size,
      size_t* comp_chunk_size) 
  {
    do_snap(
        this_decomp_buffer,
        decomp_size,
        tmp_output_buffer,
        max_comp_chunk_size,
        nullptr, // snappy status -- could add this later. Need to work through how to do error checking.
        comp_chunk_size);
  }

  __device__ nvcompStatus_t& get_output_status() final override {
    return *status;
  }

  __device__ FormatType get_format_type() final override {
    return FormatType::Snappy;
  }
};

struct snappy_decompress_wrapper : hlif_decompress_wrapper {

private:
  nvcompStatus_t* status;

public:
  __device__ snappy_decompress_wrapper(uint8_t* /*shared_buffer*/, nvcompStatus_t* status)
    : status(status)
  {}
      
  __device__ void decompress_chunk(
      uint8_t* decomp_buffer,
      const uint8_t* comp_buffer,
      const size_t comp_chunk_size,
      const size_t decomp_buffer_size) 
  {
    do_unsnap(
        comp_buffer,
        comp_chunk_size,
        decomp_buffer,
        decomp_buffer_size,
        status,
        nullptr); // device_uncompressed_bytes -- unnecessary for HLIF
  }

  __device__ nvcompStatus_t& get_output_status() final override {
    return *status;
  }
};

void snappyHlifBatchCompress(
    CommonHeader* common_header,
    const uint8_t* decomp_buffer, 
    const size_t decomp_buffer_size, 
    uint8_t* comp_buffer, 
    uint8_t* tmp_buffer,
    const size_t raw_chunk_size,
    size_t* ix_output,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t max_comp_chunk_size,
    size_t* comp_chunk_offsets,
    size_t* comp_chunk_sizes,
    const uint32_t max_ctas,
    cudaStream_t stream,
    nvcompStatus_t* output_status) 
{
  const dim3 grid(max_ctas);
  const dim3 block(COMP_THREADS_PER_BLOCK);

  HlifCompressBatchKernel<snappy_compress_wrapper><<<grid, block, 0, stream>>>(
      common_header,
      decomp_buffer, 
      decomp_buffer_size, 
      comp_buffer, 
      tmp_buffer,
      raw_chunk_size,
      ix_output,
      ix_chunk,
      num_chunks,
      max_comp_chunk_size,
      comp_chunk_offsets,
      comp_chunk_sizes,
      output_status);      
}

void snappyHlifBatchDecompress(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t raw_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    const uint32_t max_ctas,
    cudaStream_t stream,
    nvcompStatus_t* output_status) 
{
  const dim3 grid(max_ctas);
  const dim3 block(DECOMP_THREADS_PER_BLOCK);
  HlifDecompressBatchKernel<snappy_decompress_wrapper><<<grid, block, 0, stream>>>(
      comp_buffer,
      decomp_buffer,
      raw_chunk_size,
      ix_chunk,
      num_chunks,
      comp_chunk_offsets,
      comp_chunk_sizes,
      output_status);
}

size_t snappyHlifCompMaxBlockOccupancy(const int device_id) 
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int numBlocksPerSM;
  constexpr int shmem_size = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, 
      HlifCompressBatchKernel<snappy_compress_wrapper>, 
      COMP_THREADS_PER_BLOCK,
      shmem_size);
  
  return deviceProp.multiProcessorCount * numBlocksPerSM;
}

size_t snappyHlifDecompMaxBlockOccupancy(const int device_id) 
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  int numBlocksPerSM;
  constexpr int shmem_size = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, 
      HlifDecompressBatchKernel<snappy_decompress_wrapper, 1>, 
      DECOMP_THREADS_PER_BLOCK, 
      shmem_size);
  
  return deviceProp.multiProcessorCount * numBlocksPerSM;
}

} // nvcomp namespace
