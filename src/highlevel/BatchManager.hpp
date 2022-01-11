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

#include "src/Check.h"
#include "src/CudaUtils.h"
#include "src/common.h"
#include "src/highlevel/hlif_internal.hpp"

namespace nvcomp {

// In the below, need the ix_output / ix_chunk to be 
struct CompressionConfig {
  uint64_t* ix_output;
  uint32_t* ix_chunk;
    
  ~CompressionConfig() {
    cudaFreeHost(ix_output);
    cudaFreeHost(ix_chunk);
  }
};

struct DecompressionConfig {
  size_t decomp_data_size;
  uint32_t num_chunks;
  uint32_t* ix_chunk;
  
  ~DecompressionConfig() {
    cudaFreeHost(ix_chunk);
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
  cudaStream_t internal_stream;
  bool filled_tmp_buffer;
  CommonHeader* common_header_cpu;

protected:
  uint64_t* ix_output;
  uint32_t* ix_chunk;
  uint32_t max_comp_ctas;
  uint32_t max_decomp_ctas;
  cudaStream_t user_stream;
  uint8_t* tmp_buffer;
  size_t max_comp_chunk_size;
  size_t tmp_buffer_size;
  size_t uncomp_chunk_size;
  FormatSpecHeader format_spec;
  int device_id;

public: // Method definitions - ToDo: Split into public / private
  BatchManager(size_t uncomp_chunk_size, cudaStream_t user_stream = 0, int device_id = 0)
    : filled_tmp_buffer(false),
      user_stream(user_stream),
      tmp_buffer(nullptr),
      tmp_buffer_size(0),
      device_id(device_id),
      uncomp_chunk_size(uncomp_chunk_size)
  {
    cudaStreamCreate(&internal_stream);    
    
    gpuErrchk(cudaMalloc(&ix_output, sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&ix_chunk, sizeof(uint32_t)));
    
    gpuErrchk(cudaHostAlloc(&common_header_cpu, sizeof(CommonHeader), cudaHostAllocDefault));
  }

  virtual ~BatchManager() {
    gpuErrchk(cudaFree(ix_output));
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
      const uint8_t* decomp_buffer,
      const size_t decomp_buffer_size,
      uint8_t* comp_data_buffer,
      const uint32_t num_chunks,
      size_t* comp_chunk_offsets,
      size_t* comp_chunk_sizes) = 0;

  virtual void do_decompress(
      const uint8_t* comp_data_buffer,
      uint8_t* decomp_buffer,
      const uint32_t num_chunks,
      const size_t* comp_chunk_offsets,
      const size_t* comp_chunk_sizes) = 0;

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
    // Need to pad 
    void* offset_input = sizeof(FormatSpecHeader) > 0 ? reinterpret_cast<void*>(comp_format_header + 1)
                                                      : reinterpret_cast<void*>(comp_format_header);
    size_t* comp_chunk_offsets = roundUpToAlignment<size_t>(offset_input);
    size_t* comp_chunk_sizes = comp_chunk_offsets + num_chunks;
    uint32_t* comp_chunk_checksums = reinterpret_cast<uint32_t*>(comp_chunk_sizes + num_chunks);
    uint32_t* decomp_chunk_checksums = comp_chunk_checksums + num_chunks;
    uint8_t* comp_data_buffer = reinterpret_cast<uint8_t*>(decomp_chunk_checksums + num_chunks);
    uint32_t comp_data_offset = (uintptr_t)(comp_data_buffer - comp_buffer);

    // Initialize necessary values.

    auto comp_config = std::make_shared<CompressionConfig>();
    gpuErrchk(cudaHostAlloc(&comp_config->ix_output, sizeof(uint64_t), cudaHostAllocDefault));
    gpuErrchk(cudaHostAlloc(&comp_config->ix_chunk, sizeof(uint32_t), cudaHostAllocDefault));
    
    *comp_config->ix_output = 0;
    *comp_config->ix_chunk = std::min(max_comp_ctas, num_chunks);
    gpuErrchk(cudaMemcpyAsync(ix_output, comp_config->ix_output, sizeof(uint64_t), cudaMemcpyHostToDevice, user_stream));
    gpuErrchk(cudaMemcpyAsync(ix_chunk, comp_config->ix_chunk, sizeof(uint32_t), cudaMemcpyHostToDevice, user_stream));
    
    FormatSpecHeader* cpu_format_header = get_format_header();

    *common_header_cpu = CommonHeader{
        0, // magic number
        2, // major version
        2, // minor version
        0, // compressed_data_size -- fill in later
        decomp_buffer_size, // decompressed_data_size 
        num_chunks,
        true, // include chunk starts
        0, // full comp buffer checksum
        0, // full decomp buffer checksum
        true, // include_per_chunk_comp_buffer_checksums 
        true, // include_per_chunk_decomp_buffer_checksums
        uncomp_chunk_size, 
        comp_data_offset};

    gpuErrchk(cudaMemcpyAsync(common_header, common_header_cpu, sizeof(CommonHeader), cudaMemcpyHostToDevice, internal_stream));
    gpuErrchk(cudaMemcpyAsync(comp_format_header, cpu_format_header, sizeof(FormatSpecHeader), cudaMemcpyHostToDevice, internal_stream));
    
    do_compress(
        decomp_buffer,
        decomp_buffer_size,
        comp_data_buffer,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes);

    gpuErrchk(cudaMemcpyAsync(&(common_header->comp_data_size), ix_output, sizeof(uint64_t), cudaMemcpyDeviceToHost, user_stream));
    gpuErrchk(cudaStreamSynchronize(internal_stream));
    return comp_config;
  }

  std::shared_ptr<DecompressionConfig> 
  configure_decompression(uint8_t* comp_buffer) final override
  {
    // To do this, need synchronous memcpy because 
    // the user will need to allocate an output buffer based on the result
    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    auto decomp_config = std::make_shared<DecompressionConfig>();
    gpuErrchk(cudaMemcpyAsync(&decomp_config->decomp_data_size, 
        &common_header->decomp_data_size, 
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        internal_stream));

    gpuErrchk(cudaMemcpyAsync(&decomp_config->num_chunks, 
        &common_header->num_chunks, 
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        internal_stream));

    gpuErrchk(cudaHostAlloc(&decomp_config->ix_chunk, sizeof(uint32_t), cudaHostAllocDefault));

    gpuErrchk(cudaStreamSynchronize(internal_stream));

    return decomp_config;
  }

  size_t get_compressed_output_size(uint8_t* comp_buffer) final override
  {
    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    
    gpuErrchk(cudaMemcpyAsync(common_header_cpu, 
        common_header, 
        sizeof(CommonHeader),
        cudaMemcpyDeviceToHost,
        internal_stream));
    gpuErrchk(cudaStreamSynchronize(internal_stream));
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

    *config.ix_chunk = std::min(max_decomp_ctas * get_decomp_chunks_per_block(), config.num_chunks);
    gpuErrchk(cudaMemcpyAsync(ix_chunk, config.ix_chunk, sizeof(uint32_t), cudaMemcpyHostToDevice, user_stream));
    
    do_decompress(
        comp_data_buffer,
        decomp_buffer,
        config.num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes);
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