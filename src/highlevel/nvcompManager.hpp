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
#include "PinnedPtrs.hpp"

namespace nvcomp {

struct CompressionConfig {
private: 
  std::shared_ptr<PinnedPtrWrapper<nvcompStatus_t>> status;

public:
  size_t max_compressed_buffer_size;

  CompressionConfig(
      PinnedPtrPool<nvcompStatus_t>& pool, 
      size_t max_compressed_buffer_size)
    : status(std::make_shared<PinnedPtrWrapper<nvcompStatus_t>>(pool)),
      max_compressed_buffer_size(max_compressed_buffer_size)
  {
    *get_status() = nvcompSuccess;
  }

  nvcompStatus_t* get_status() const {
    return status->ptr;
  }
};

struct DecompressionConfig {
private: 
  std::shared_ptr<PinnedPtrWrapper<nvcompStatus_t>> status;

public:
  size_t decomp_data_size;
  uint32_t num_chunks;

  DecompressionConfig(PinnedPtrPool<nvcompStatus_t>& pool)
    : status(std::make_shared<PinnedPtrWrapper<nvcompStatus_t>>(pool)),
      decomp_data_size(),
      num_chunks()
  {
    *get_status() = nvcompSuccess;
  }

  nvcompStatus_t* get_status() const {
    return status->ptr;
  }
};

struct nvcompManagerBase {
  virtual void compress(
      const uint8_t* decomp_buffer, 
      const size_t decomp_buffer_size, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) = 0;

  virtual CompressionConfig configure_compression(const size_t decomp_buffer_size) = 0;

  virtual DecompressionConfig configure_decompression(uint8_t* comp_buffer) = 0;

  virtual void decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config) = 0;
  
  virtual void set_scratch_buffer(uint8_t* new_scratch_buffer) = 0;

  virtual size_t get_scratch_buffer_size() = 0;
  
  virtual size_t get_compressed_output_size(uint8_t* comp_buffer) = 0;

  virtual ~nvcompManagerBase() = default;
};

template <typename FormatSpecHeader>
struct ManagerBase : nvcompManagerBase {

protected: // members
  CommonHeader* common_header_cpu;
  cudaStream_t user_stream;
  uint8_t* scratch_buffer;
  size_t scratch_buffer_size;
  int device_id;
  PinnedPtrPool<nvcompStatus_t> status_pool;
  bool manager_filled_scratch_buffer;

private: // members
  bool scratch_buffer_filled;

public: // API
  ManagerBase(cudaStream_t user_stream = 0, int device_id = 0) 
    : common_header_cpu(),
      user_stream(user_stream),
      scratch_buffer(nullptr),
      scratch_buffer_size(0),
      device_id(device_id),
      status_pool(),
      manager_filled_scratch_buffer(false),
      scratch_buffer_filled(false)
  {
    gpuErrchk(cudaHostAlloc(&common_header_cpu, sizeof(CommonHeader), cudaHostAllocDefault));
  }

  size_t get_scratch_buffer_size() final override {
    return scratch_buffer_size;
  }

  ManagerBase(const ManagerBase&) = delete;
  ManagerBase& operator=(const ManagerBase&) = delete;

  size_t get_compressed_output_size(uint8_t* comp_buffer) final override {
    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    
    gpuErrchk(cudaMemcpyAsync(common_header_cpu, 
        common_header, 
        sizeof(CommonHeader),
        cudaMemcpyDefault,
        user_stream));
    gpuErrchk(cudaStreamSynchronize(user_stream));
    return common_header_cpu->comp_data_size + common_header_cpu->comp_data_offset;
  };
  
  virtual ~ManagerBase() {
    gpuErrchk(cudaFreeHost(common_header_cpu));
    if (manager_filled_scratch_buffer) {
      #if CUDART_VERSION >= 11020
        gpuErrchk(cudaFreeAsync(scratch_buffer, user_stream));
      #else 
        gpuErrchk(cudaFree(scratch_buffer));
      #endif
    }
  }

  CompressionConfig configure_compression(const size_t decomp_buffer_size) final override
  {
    const size_t max_comp_size = calculate_max_compressed_output_size(decomp_buffer_size);
    return CompressionConfig{status_pool, max_comp_size};
  }

  DecompressionConfig configure_decompression(uint8_t* comp_buffer) final override
  {
    // To do this, need synchronous memcpy because 
    // the user will need to allocate an output buffer based on the result
    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    DecompressionConfig decomp_config{status_pool};
    
    gpuErrchk(cudaMemcpy(&decomp_config.decomp_data_size, 
        &common_header->decomp_data_size, 
        sizeof(size_t),
        cudaMemcpyDefault));

    gpuErrchk(cudaMemcpy(&decomp_config.num_chunks, 
        &common_header->num_chunks, 
        sizeof(size_t),
        cudaMemcpyDefault));

    return decomp_config;
  }

  void set_scratch_buffer(uint8_t* new_scratch_buffer) final override
  {
    // TODO: What if the temp buffer is already set? Part of reconfiguration idea
    if (scratch_buffer_filled) return;
    scratch_buffer = new_scratch_buffer;
    scratch_buffer_filled = true;
  }

  virtual void compress(
      const uint8_t* decomp_buffer, 
      const size_t decomp_buffer_size, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) 
  {
    if (not scratch_buffer_filled) {
      #if CUDART_VERSION >= 11020
        gpuErrchk(cudaMallocAsync(&scratch_buffer, scratch_buffer_size, user_stream));
      #else
        gpuErrchk(cudaMalloc(&scratch_buffer, scratch_buffer_size));
      #endif
      scratch_buffer_filled = true;
      manager_filled_scratch_buffer = true;
    }    

    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    FormatSpecHeader* comp_format_header = reinterpret_cast<FormatSpecHeader*>(common_header + 1);
    gpuErrchk(cudaMemcpyAsync(comp_format_header, get_format_header(), sizeof(FormatSpecHeader), cudaMemcpyDefault, user_stream));

    gpuErrchk(cudaMemsetAsync(&common_header->comp_data_size, 0, sizeof(uint64_t), user_stream));

    uint8_t* new_comp_buffer = comp_buffer + sizeof(CommonHeader) + sizeof(FormatSpecHeader);
    do_compress(common_header, decomp_buffer, decomp_buffer_size, new_comp_buffer, comp_config);
  }

  virtual void decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config)
  {
    const uint8_t* new_comp_buffer = comp_buffer + sizeof(CommonHeader) + sizeof(FormatSpecHeader);

    do_decompress(decomp_buffer, new_comp_buffer, config);
  }
  
protected: // helpers 
  void finish_init() {
    scratch_buffer_size = compute_scratch_buffer_size();
  }

private: // helpers
  virtual void do_compress(
      CommonHeader* common_header,
      const uint8_t* decomp_buffer, 
      const size_t decomp_buffer_size, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) = 0;

  virtual void do_decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config) = 0;

  virtual size_t compute_scratch_buffer_size() = 0;

  virtual size_t calculate_max_compressed_output_size(size_t decomp_buffer_size) = 0;

  virtual FormatSpecHeader* get_format_header() = 0;
};

} // namespace nvcomp