#pragma once

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>
#include <vector>

#include "src/Check.h"
#include "src/CudaUtils.h"
#include "src/common.h"
#include "nvcomp_common_deps/hlif_shared_types.h"

namespace nvcomp {

template<typename T>
struct PinnedPtrPool {
  const static int POOL_PREALLOC_SIZE = 10;
  const static int POOL_REALLOC_SIZE = 5;
  std::vector<T*> pool;
  std::vector<T*> alloced_buffers; 

  PinnedPtrPool() 
    : alloced_buffers(1)
  {
    T*& first_alloc = alloced_buffers[0];

    pool.reserve(POOL_PREALLOC_SIZE);

    gpuErrchk(cudaHostAlloc(&first_alloc, POOL_PREALLOC_SIZE * sizeof(T), cudaHostAllocDefault));

    for (size_t ix; ix < POOL_PREALLOC_SIZE; ++ix) {
      pool.push_back(first_alloc + ix);
    }
  }

  void push_ptr(T* status) 
  {
    pool.push_back(status);
  }

  T* pop_status() 
  {
    if (pool.empty()) {
      // realloc
      alloced_buffers.push_back(nullptr);
      T*& new_alloc = alloced_buffers.back();

      gpuErrchk(cudaHostAlloc(&new_alloc, POOL_REALLOC_SIZE * sizeof(T), cudaHostAllocDefault));
      for (size_t ix; ix < POOL_REALLOC_SIZE; ++ix) {
        pool.push_back(new_alloc + ix);
      }
    } 

    T* res = pool.back();
    pool.pop_back();
    return res;
  }

  ~PinnedPtrPool() {
    for (auto alloced_buffer : alloced_buffers) {
      gpuErrchk(cudaFreeHost(alloced_buffer));
    }
  }
};

// In the below, need the ix_output / ix_chunk to be 
struct CompressionConfig {
  nvcompStatus_t* output_status;
  PinnedPtrPool<nvcompStatus_t>& status_pool;

  CompressionConfig(PinnedPtrPool<nvcompStatus_t>& pool)
    : output_status(pool.pop_status()),
      status_pool(pool)
  {}

  ~CompressionConfig() {
    status_pool.push_ptr(output_status);
  }
};

struct DecompressionConfig {
  size_t decomp_data_size;
  uint32_t num_chunks;
  nvcompStatus_t* output_status;
  PinnedPtrPool<nvcompStatus_t>& status_pool;

  DecompressionConfig(PinnedPtrPool<nvcompStatus_t>& pool)
    : output_status(pool.pop_status()),
      status_pool(pool)
  {}

  ~DecompressionConfig() {
    status_pool.push_ptr(output_status);
  }
};

struct BatchManagerBase {
  virtual std::shared_ptr<CompressionConfig> compress(
      const uint8_t* decomp_buffer, 
      const size_t decomp_buffer_size, 
      uint8_t* comp_buffer) = 0;

  virtual std::shared_ptr<DecompressionConfig> configure_decompression(uint8_t* comp_buffer) = 0;

  virtual void decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config) = 0;
  
  virtual void set_tmp_buffer(uint8_t* new_tmp_buffer) = 0;

  virtual size_t calculate_max_compressed_output_size(size_t decomp_buffer_size) = 0;
  
  virtual size_t get_tmp_buffer_size() = 0;
  
  virtual size_t get_compressed_output_size(uint8_t* comp_buffer) = 0;
};

template<typename FormatSpecHeader>
struct BatchManager : BatchManagerBase {

private: // typedefs

private: // members
  bool filled_tmp_buffer;

protected:
  uint32_t* ix_chunk;
  uint32_t max_comp_ctas;
  uint32_t max_decomp_ctas;
  cudaStream_t user_stream;
  uint8_t* tmp_buffer;
  size_t max_comp_chunk_size;
  size_t tmp_buffer_size;
  size_t uncomp_chunk_size;
  FormatSpecHeader format_spec;
  CommonHeader* common_header_cpu;
  int device_id;
  PinnedPtrPool<nvcompStatus_t> status_pool;

public: // Method definitions - ToDo: Split into public / private
  BatchManager(size_t uncomp_chunk_size, cudaStream_t user_stream = 0, int device_id = 0)
    : filled_tmp_buffer(false),
      user_stream(user_stream),
      tmp_buffer(nullptr),
      tmp_buffer_size(0),
      device_id(device_id),
      uncomp_chunk_size(uncomp_chunk_size),
      status_pool()
  {
    gpuErrchk(cudaMalloc(&ix_chunk, sizeof(uint32_t)));
    gpuErrchk(cudaHostAlloc(&common_header_cpu, sizeof(CommonHeader), cudaHostAllocDefault));
  }

  virtual ~BatchManager() {
    gpuErrchk(cudaFree(ix_chunk));
    gpuErrchk(cudaFreeHost(common_header_cpu));
  }

  void finish_init() {
    max_comp_ctas = compute_compression_max_block_occupancy();
    max_decomp_ctas = compute_decompression_max_block_occupancy();
    tmp_buffer_size = compute_tmp_buffer_size();
  }

  virtual size_t compute_max_compressed_chunk_size() = 0;
  virtual uint32_t compute_compression_max_block_occupancy() = 0;
  virtual uint32_t compute_decompression_max_block_occupancy() = 0;
  virtual size_t compute_tmp_buffer_size() = 0;
  virtual FormatSpecHeader* get_format_header() = 0;

  virtual void do_compress(
      CommonHeader* common_header,
      const uint8_t* decomp_buffer,
      const size_t decomp_buffer_size,
      uint8_t* comp_data_buffer,
      const uint32_t num_chunks,
      size_t* comp_chunk_offsets,
      size_t* comp_chunk_sizes,
      nvcompStatus_t* output_status) = 0;

  virtual void do_decompress(
      const uint8_t* comp_data_buffer,
      uint8_t* decomp_buffer,
      const uint32_t num_chunks,
      const size_t* comp_chunk_offsets,
      const size_t* comp_chunk_sizes,
      nvcompStatus_t* output_status) = 0;

  virtual uint8_t get_decomp_chunks_per_block()
  {
    return 1;
  }

  std::shared_ptr<CompressionConfig> compress(
      const uint8_t* decomp_buffer, 
      const size_t decomp_buffer_size, 
      uint8_t* comp_buffer) final override
  {
    if (not filled_tmp_buffer) {
      gpuErrchk(cudaMallocAsync(&tmp_buffer, tmp_buffer_size, user_stream));
      filled_tmp_buffer = true;
    }
    
    const uint32_t num_chunks = roundUpDiv(decomp_buffer_size, uncomp_chunk_size);
    
    // Set up the raw pointers
    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    FormatSpecHeader* comp_format_header = reinterpret_cast<FormatSpecHeader*>(common_header + 1);
    // Pad so that the comp chunk offsets are properly aligned
    void* offset_input = sizeof(FormatSpecHeader) > 0 ? reinterpret_cast<void*>(comp_format_header + 1)
                                                      : reinterpret_cast<void*>(comp_format_header);
    size_t* comp_chunk_offsets = roundUpToAlignment<size_t>(offset_input);
    size_t* comp_chunk_sizes = comp_chunk_offsets + num_chunks;
    uint32_t* comp_chunk_checksums = reinterpret_cast<uint32_t*>(comp_chunk_sizes + num_chunks);
    uint32_t* decomp_chunk_checksums = comp_chunk_checksums + num_chunks;
    uint8_t* comp_data_buffer = reinterpret_cast<uint8_t*>(decomp_chunk_checksums + num_chunks);
    uint32_t comp_data_offset = (uintptr_t)(comp_data_buffer - comp_buffer);

    // Initialize necessary values.
    auto comp_config = std::make_shared<CompressionConfig>(status_pool);

    gpuErrchk(cudaMemsetAsync(&common_header->comp_data_size, 0, sizeof(uint64_t), user_stream));
    gpuErrchk(cudaMemsetAsync(ix_chunk, 0, sizeof(uint32_t), user_stream));
    
    FormatSpecHeader* cpu_format_header = get_format_header();
    gpuErrchk(cudaMemcpyAsync(comp_format_header, cpu_format_header, sizeof(FormatSpecHeader), cudaMemcpyHostToDevice, user_stream));
    
    do_compress(
        common_header,
        decomp_buffer,
        decomp_buffer_size,
        comp_data_buffer,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        comp_config->output_status);
    
    return comp_config;
  }

  std::shared_ptr<DecompressionConfig> 
  configure_decompression(uint8_t* comp_buffer) final override
  {
    // To do this, need synchronous memcpy because 
    // the user will need to allocate an output buffer based on the result
    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    auto decomp_config = std::make_shared<DecompressionConfig>(status_pool);
    gpuErrchk(cudaMemcpyAsync(&decomp_config->decomp_data_size, 
        &common_header->decomp_data_size, 
        sizeof(size_t),
        cudaMemcpyDefault,
        user_stream));

    gpuErrchk(cudaMemcpyAsync(&decomp_config->num_chunks, 
        &common_header->num_chunks, 
        sizeof(size_t),
        cudaMemcpyDefault,
        user_stream));

    *decomp_config->output_status = nvcompSuccess;

    return decomp_config;
  }

  size_t get_compressed_output_size(uint8_t* comp_buffer) final override
  {
    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    
    gpuErrchk(cudaMemcpyAsync(common_header_cpu, 
        common_header, 
        sizeof(CommonHeader),
        cudaMemcpyDeviceToHost,
        user_stream));
    gpuErrchk(cudaStreamSynchronize(user_stream));
    return common_header_cpu->comp_data_size + common_header_cpu->comp_data_offset;
  }

  size_t get_tmp_buffer_size() final override {
    return tmp_buffer_size;
  }

  void decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config) final override
  {
    const CommonHeader* common_header = reinterpret_cast<const CommonHeader*>(comp_buffer);
    const FormatSpecHeader* comp_format_header = reinterpret_cast<const FormatSpecHeader*>(common_header + 1);
    const void* offset_input = sizeof(FormatSpecHeader) > 0 ? reinterpret_cast<const void*>(comp_format_header + 1)
                                                            : reinterpret_cast<const void*>(comp_format_header);
    const size_t* comp_chunk_offsets = roundUpToAlignment<const size_t>(offset_input);
    const size_t* comp_chunk_sizes = comp_chunk_offsets + config.num_chunks;
    const uint32_t* comp_chunk_checksums = reinterpret_cast<const uint32_t*>(comp_chunk_sizes + config.num_chunks);
    const uint32_t* decomp_chunk_checksums = comp_chunk_checksums + config.num_chunks;
    const uint8_t* comp_data_buffer = reinterpret_cast<const uint8_t*>(decomp_chunk_checksums + config.num_chunks);

    gpuErrchk(cudaMemsetAsync(ix_chunk, 0, sizeof(uint32_t), user_stream));
    do_decompress(
        comp_data_buffer,
        decomp_buffer,
        config.num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        config.output_status);
  }

  size_t calculate_max_compressed_output_size(size_t decomp_buffer_size) final override
  {
    const size_t num_chunks = roundUpDiv(decomp_buffer_size, uncomp_chunk_size);

    const size_t comp_buffer_size = max_comp_chunk_size * num_chunks;

    const size_t chunk_offsets_size = sizeof(ChunkStartOffset_t) * num_chunks;
    const size_t chunk_sizes_size = sizeof(uint32_t) * num_chunks;
    // *2 for decomp and comp checksums
    const size_t checksum_size = sizeof(Checksum_t) * num_chunks * 2;

    return sizeof(CommonHeader) + sizeof(FormatSpecHeader) + 
        chunk_offsets_size + chunk_sizes_size + checksum_size + comp_buffer_size;
  }

  void set_tmp_buffer(uint8_t* new_tmp_buffer) final override
  {
    // TODO: What if the temp buffer is already set? Part of reconfiguration idea
    if (filled_tmp_buffer) return;

    tmp_buffer = new_tmp_buffer;
    filled_tmp_buffer = true;
  }

};

} // namespace nvcomp