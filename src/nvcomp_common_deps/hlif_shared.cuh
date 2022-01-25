/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cassert>

#include "shared_types.h"
#include "hlif_shared_types.h"

// Compress wrapper must meet this requirement
struct hlif_compress_wrapper {
  virtual __device__ void compress_chunk(
      uint8_t* tmp_output_buffer,
      const uint8_t* this_decomp_buffer,
      const size_t decomp_size,
      const size_t max_comp_chunk_size, 
      size_t* comp_chunk_size) = 0;
  
  virtual __device__ nvcompStatus_t& get_output_status() = 0;

  virtual __device__ FormatType get_format_type() = 0;

  virtual __device__ ~hlif_compress_wrapper() {};
};

// Decompress wrapper must meet this requirement
struct hlif_decompress_wrapper {
  virtual __device__ void decompress_chunk(
      uint8_t* decomp_buffer,
      const uint8_t* comp_buffer,
      const size_t comp_chunk_size,
      const size_t decomp_buffer_size) = 0;
      
  virtual __device__ nvcompStatus_t& get_output_status() = 0;
  
  virtual __device__ ~hlif_decompress_wrapper() {};
};

__device__ inline void fill_common_header(
    const uint8_t* comp_buffer, 
    CommonHeader* common_header,
    const size_t decomp_buffer_size, 
    const size_t raw_chunk_size,
    const size_t num_chunks,
    const FormatType format_type) 
{
  common_header->magic_number = 0;
  common_header->major_version = 2;
  common_header->minor_version = 2;
  common_header->format = format_type;
  common_header->decomp_data_size = decomp_buffer_size;
  common_header->num_chunks = num_chunks;
  common_header->include_chunk_starts = true;
  common_header->full_comp_buffer_checksum = 0;
  common_header->decomp_buffer_checksum = 0;
  common_header->include_per_chunk_comp_buffer_checksums = false;
  common_header->include_per_chunk_decomp_buffer_checksums = false;
  common_header->uncomp_chunk_size = raw_chunk_size;
  common_header->comp_data_offset = (uintptr_t)comp_buffer - (uintptr_t)common_header;
}

__device__ inline void copyTmpBuffer(
    size_t* comp_chunk_offsets,
    size_t* comp_chunk_sizes,
    const uint8_t* tmp_output_buffer,
    uint8_t* comp_buffer,
    uint64_t* ix_output,
    uint32_t ix_chunk)
{
  // Do the copy into the final buffer.
  size_t comp_chunk_offset = comp_chunk_offsets[ix_chunk];
  size_t comp_chunk_size = comp_chunk_sizes[ix_chunk];
  const int ix_alignment_input = sizeof(uint32_t) - ((uintptr_t)tmp_output_buffer % sizeof(uint32_t));
  if (ix_alignment_input % 4 == 0) {
    const char4* aligned_input = reinterpret_cast<const char4*>(tmp_output_buffer);
    uint8_t* output = comp_buffer + comp_chunk_offset;
    for (size_t ix = threadIdx.x; ix < comp_chunk_size / 4; ix += blockDim.x) {
      char4 val = aligned_input[ix];
      output[4 * ix] = val.x;
      output[4 * ix + 1] = val.y;
      output[4 * ix + 2] = val.z;
      output[4 * ix + 3] = val.w;
    }
    int rem_bytes = comp_chunk_size % sizeof(uint32_t);
    if (threadIdx.x < rem_bytes) {
      output[comp_chunk_size - rem_bytes + threadIdx.x] = tmp_output_buffer[comp_chunk_size - rem_bytes + threadIdx.x];
    }
  } else {
    for (size_t ix = threadIdx.x; ix < comp_chunk_size; ix += blockDim.x) {
      comp_buffer[comp_chunk_offset + ix] = tmp_output_buffer[ix];
    }
  }
}

template<typename CompressT>
__device__ inline void HlifCompressBatch(
  const uint8_t* decomp_buffer, 
    const size_t decomp_buffer_size, 
    uint8_t* comp_buffer, 
    uint8_t* tmp_buffer,
    const size_t raw_chunk_size,
    uint64_t* ix_output,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t max_comp_chunk_size,
    size_t* comp_chunk_offsets,
    size_t* comp_chunk_sizes,
    uint8_t* share_buffer,
    nvcompStatus_t* kernel_output_status,
    CompressT&& compressor)
{
  __shared__ uint32_t this_ix_chunk;

  if (threadIdx.x == 0) {
    this_ix_chunk = blockIdx.x;
  }

  uint8_t* tmp_output_buffer = tmp_buffer + blockIdx.x * max_comp_chunk_size;  
  __syncthreads();
  
  int initial_chunks = gridDim.x;  

  while (this_ix_chunk < num_chunks) {
    size_t ix_decomp_start = this_ix_chunk * raw_chunk_size;
    const uint8_t* this_decomp_buffer = decomp_buffer + ix_decomp_start;
    size_t decomp_size = min(raw_chunk_size, decomp_buffer_size - ix_decomp_start);
    compressor.compress_chunk(
        tmp_output_buffer,
        this_decomp_buffer,
        decomp_size,
        max_comp_chunk_size,
        &comp_chunk_sizes[this_ix_chunk]);

    // Determine the right place to output this buffer.
    if (threadIdx.x == 0) {
      static_assert(sizeof(uint64_t) == sizeof(unsigned long long int));
      comp_chunk_offsets[this_ix_chunk] = atomicAdd(
          reinterpret_cast<unsigned long long int*>(ix_output), 
          comp_chunk_sizes[this_ix_chunk]);
    }

    __syncthreads();

    copyTmpBuffer(
        comp_chunk_offsets,
        comp_chunk_sizes,
        tmp_output_buffer,
        comp_buffer,
        ix_output,
        this_ix_chunk);

    if (threadIdx.x == 0) {
      this_ix_chunk = initial_chunks + atomicAdd(ix_chunk, size_t{1});
    }
    __syncthreads();
  }

  // Check for errors. Any error should be reported in the global status value
  if (threadIdx.x == 0) {
    if (compressor.get_output_status() != nvcompSuccess) {
      *kernel_output_status = compressor.get_output_status();
    }
  }
}

template<typename CompressT, 
         typename CompArg>
__global__ std::enable_if_t<std::is_base_of<hlif_compress_wrapper, CompressT>::value>
HlifCompressBatchKernel(
    CommonHeader* common_header,
    const uint8_t* decomp_buffer, 
    const size_t decomp_buffer_size, 
    uint8_t* comp_buffer, 
    uint8_t* tmp_buffer,
    const size_t raw_chunk_size,
    uint64_t* ix_output,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t max_comp_chunk_size,
    size_t* comp_chunk_offsets,
    size_t* comp_chunk_sizes,
    nvcompStatus_t* kernel_output_status,
    CompArg compress_arg)
{
  extern __shared__ uint8_t share_buffer[];

  uint8_t* free_tmp_buffer = tmp_buffer + max_comp_chunk_size * gridDim.x;
  
  __shared__ nvcompStatus_t output_status;
  
  CompressT compressor{compress_arg, free_tmp_buffer, share_buffer, &output_status};
  if (blockIdx.x == 0 and threadIdx.x == 0) {
    fill_common_header(comp_buffer, common_header, decomp_buffer_size, raw_chunk_size, num_chunks, compressor.get_format_type());
  }
  
  HlifCompressBatch(
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
      share_buffer,
      kernel_output_status,
      compressor);
}

template<typename CompressT>
__global__ std::enable_if_t<std::is_base_of<hlif_compress_wrapper, CompressT>::value>
HlifCompressBatchKernel(
    CommonHeader* common_header,
    const uint8_t* decomp_buffer, 
    const size_t decomp_buffer_size, 
    uint8_t* comp_buffer, 
    uint8_t* tmp_buffer,
    const size_t raw_chunk_size,
    uint64_t* ix_output,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t max_comp_chunk_size,
    size_t* comp_chunk_offsets,
    size_t* comp_chunk_sizes,
    nvcompStatus_t* kernel_output_status)
{
  extern __shared__ uint8_t share_buffer[];
  
  uint8_t* free_tmp_buffer = tmp_buffer + max_comp_chunk_size * gridDim.x;
  __shared__ nvcompStatus_t output_status;

  CompressT compressor{free_tmp_buffer, share_buffer, &output_status};
  
  if (blockIdx.x == 0 and threadIdx.x == 0) {
    fill_common_header(comp_buffer, common_header, decomp_buffer_size, raw_chunk_size, num_chunks, compressor.get_format_type());
  }
  
  HlifCompressBatch(
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
      share_buffer,
      kernel_output_status,
      compressor);
}

template<typename DecompressT,
         int chunks_per_block>
__device__ inline void HlifDecompressBatch(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t raw_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    uint8_t* share_buffer,
    nvcompStatus_t* kernel_output_status,
    DecompressT decompressor)
{
  bool use_warp_sync = blockDim.x == 32;
  if (not use_warp_sync) {
    assert(blockDim.y == 1);
  }

  assert(chunks_per_block == blockDim.y);
  
  __shared__ uint32_t ix_chunks[chunks_per_block];

  volatile uint32_t& this_ix_chunk = *(ix_chunks + threadIdx.y);
  if (threadIdx.x == 0) {
    this_ix_chunk = blockIdx.x * chunks_per_block + threadIdx.y;
  }

  if (use_warp_sync) {
    __syncwarp();
  } else {
    __syncthreads();
  }

  int initial_chunks = gridDim.x * chunks_per_block;  
  while (this_ix_chunk < num_chunks) {
    const uint8_t* this_comp_buffer = comp_buffer + comp_chunk_offsets[this_ix_chunk];
    uint8_t* this_decomp_buffer = decomp_buffer + this_ix_chunk * raw_chunk_size;

    decompressor.decompress_chunk(
        this_decomp_buffer,
        this_comp_buffer,
        comp_chunk_sizes[this_ix_chunk],
        raw_chunk_size); 

    if (threadIdx.x == 0) {
      this_ix_chunk = initial_chunks + atomicAdd(ix_chunk, uint32_t{1});
    }

    if (use_warp_sync) {
      __syncwarp();
    } else {
      __syncthreads();
    }
  }

  // Check for errors. Any error should be reported in the global status value
  if (threadIdx.x == 0) {
    if (decompressor.get_output_status() != nvcompSuccess) {
      *kernel_output_status = decompressor.get_output_status();
    }
  }
}    

template<typename DecompressT,
         int chunks_per_block = 1,
         typename DecompArg>
__global__ std::enable_if_t<std::is_base_of<hlif_decompress_wrapper, DecompressT>::value> 
HlifDecompressBatchKernel(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t raw_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    nvcompStatus_t* kernel_output_status,
    DecompArg decompress_arg)
{
  extern __shared__ uint8_t share_buffer[];
  __shared__ nvcompStatus_t output_status[chunks_per_block];
  DecompressT decompressor{decompress_arg, share_buffer, &output_status[threadIdx.y]};

  HlifDecompressBatch<DecompressT, chunks_per_block>(
      comp_buffer, 
      decomp_buffer, 
      raw_chunk_size,
      ix_chunk,
      num_chunks,
      comp_chunk_offsets,
      comp_chunk_sizes,
      share_buffer,
      kernel_output_status,
      decompressor);
}

template<typename DecompressT,
         int chunks_per_block = 1>
__global__ std::enable_if_t<std::is_base_of<hlif_decompress_wrapper, DecompressT>::value> 
HlifDecompressBatchKernel(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t raw_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    nvcompStatus_t* kernel_output_status)
{
  extern __shared__ uint8_t share_buffer[];
  __shared__ nvcompStatus_t output_status[chunks_per_block];
  DecompressT decompressor{share_buffer, &output_status[threadIdx.y]};

  HlifDecompressBatch<DecompressT, chunks_per_block>(
      comp_buffer, 
      decomp_buffer, 
      raw_chunk_size,
      ix_chunk,
      num_chunks,
      comp_chunk_offsets,
      comp_chunk_sizes,
      share_buffer,
      kernel_output_status,
      decompressor);
}
