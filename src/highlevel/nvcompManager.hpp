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

/******************************************************************************
 * CLASSES ********************************************************************
 *****************************************************************************/

/**
 * @brief Config used to aggregate information about a particular compression.
 * 
 * Contains a "PinnedPtrWrapper" to an nvcompStatus. After the compression is complete,
 * the user can check the result status which resides in pinned host memory.
 */
struct CompressionConfig {
private: 
  std::shared_ptr<PinnedPtrWrapper<nvcompStatus_t>> status;

public:
  size_t max_compressed_buffer_size;

  /**
   * @brief Construct the config given an nvcompStatus_t memory pool
   */
  CompressionConfig(
      PinnedPtrPool<nvcompStatus_t>& pool, 
      size_t max_compressed_buffer_size)
    : status(std::make_shared<PinnedPtrWrapper<nvcompStatus_t>>(pool)),
      max_compressed_buffer_size(max_compressed_buffer_size)
  {
    *get_status() = nvcompSuccess;
  }

  /**
   * @brief Get the raw nvcompStatus_t*
   */
  nvcompStatus_t* get_status() const {
    return status->ptr;
  }
};

/**
 * @brief Config used to aggregate information about a particular decompression.
 * 
 * Contains a "PinnedPtrWrapper" to an nvcompStatus. After the decompression is complete,
 * the user can check the result status which resides in pinned host memory.
 */
struct DecompressionConfig {
private: 
  std::shared_ptr<PinnedPtrWrapper<nvcompStatus_t>> status;

public:
  size_t decomp_data_size;
  uint32_t num_chunks;

  /**
   * @brief Construct the config given an nvcompStatus_t memory pool
   */
  DecompressionConfig(PinnedPtrPool<nvcompStatus_t>& pool)
    : status(std::make_shared<PinnedPtrWrapper<nvcompStatus_t>>(pool)),
      decomp_data_size(),
      num_chunks()
  {
    *get_status() = nvcompSuccess;
  }

  /**
   * @brief Get the raw nvcompStatus_t*
   */
  nvcompStatus_t* get_status() const {
    return status->ptr;
  }
};

/**
 * @brief Abstract base class that defines the nvCOMP high level interface
 */
struct nvcompManagerBase {
  /**
   * @brief Configure the compression. 
   *
   * This routine computes the size of the required result buffer. The result config also
   * contains the nvcompStatus* that allows error checking. 
   * 
   * @param decomp_buffer_size The uncompressed input data size.
   * \return comp_config Result
   */
  virtual CompressionConfig configure_compression(const size_t decomp_buffer_size) = 0;

  /**
   * @brief Perform compression asynchronously.
   *
   * @param decomp_buffer The uncompressed input data (GPU accessible).
   * @param decomp_buffer_size The length of the uncompressed input data.
   * @param comp_buffer The location to output the compressed data to (GPU accessible).
   * @param comp_config Resulted from configure_compression given this decomp_buffer_size.
   * Contains the nvcompStatus* that allows error checking. 
   */
  virtual void compress(
      const uint8_t* decomp_buffer, 
      const size_t decomp_buffer_size, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) = 0;

  /**
   * @brief Configure the decompression. 
   *
   * Initiates synchronous mem copies to retrieve blocking information
   * from the compressed device buffer. In the base case, this is just the size of the compressed buffer. 
   * 
   * @param comp_buffer The compressed input data (GPU accessible).
   * \return decomp_config Result
   */
  virtual DecompressionConfig configure_decompression(uint8_t* comp_buffer) = 0;

  /**
   * @brief Perform decompression asynchronously.
   *
   * @param decomp_buffer The location to output the decompressed data to (GPU accessible).
   * @param comp_buffer The compressed input data (GPU accessible).
   * @param decomp_config Resulted from configure_decompression given this decomp_buffer_size.
   * Contains nvcompStatus* in pinned host memory to allow error checking.
   */
  virtual void decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& decomp_config) = 0;
  
  /**
   * @brief Allows the user to provide a user-allocated scratch buffer.
   * 
   * If this routine is not called before compression / decompression is called, the manager
   * allocates the required scratch buffer. If this is called after the manager has allocated a 
   * scratch buffer, the manager frees the scratch buffer it allocated then switches to use 
   * the new user-provided one.
   * 
   * @param new_scratch_buffer The location (GPU accessible) to use for comp/decomp scratch space
   * 
   */
  virtual void set_scratch_buffer(uint8_t* new_scratch_buffer) = 0;

  /** 
   * @brief Computes the size of the required scratch space
   * 
   * This scratch space size is constant and based on the configuration of the manager and the 
   * maximum occupancy on the device.
   * 
   * \return The required scratch buffer size
   */ 
  virtual size_t get_required_scratch_buffer_size() = 0;
  
  /** 
   * @brief Computes the compressed output size of a given buffer 
   * 
   * Synchronously copies the size of the compressed buffer to a stack variable for return.
   * 
   * @param comp_buffer The start pointer of the compressed buffer to assess.
   * \return Size of the compressed buffer
   */ 
  virtual size_t get_compressed_output_size(uint8_t* comp_buffer) = 0;

  virtual ~nvcompManagerBase() = default;
};

/**
 * @brief ManagerBase contains shared functionality amongst the different nvcompManager types
 * 
 * - Intended that all Managers will inherit from this class directly or indirectly.
 *
 * - Contains a pinned memory pool for result statuses to avoid repeated 
 *   cudaHostAlloc calls on subsequent compressions / decompressions.
 * 
 * - Templated on the particular format's FormatSpecHeader so that some operations can be shared here. 
 *   This is likely to be inherited by template classes. In this case, 
 *   some usage trickery is suggested to get around dependent name lookup issues.
 *   https://en.cppreference.com/w/cpp/language/dependent_name
 *   
 */  
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
  /**
   * @brief Construct a ManagerBase
   * 
   * @param user_stream The stream to use for all operations. Optional, defaults to the default stream
   * @param device_id The default device ID to use for all operations. Optional, defaults to the default device
   */
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

  size_t get_required_scratch_buffer_size() final override {
    return scratch_buffer_size;
  }

  // Disable copying
  ManagerBase(const ManagerBase&) = delete;
  ManagerBase& operator=(const ManagerBase&) = delete;

  size_t get_compressed_output_size(uint8_t* comp_buffer) final override {
    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    
    gpuErrchk(cudaMemcpy(common_header_cpu, 
        common_header, 
        sizeof(CommonHeader),
        cudaMemcpyDefault));

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

  virtual DecompressionConfig configure_decompression(uint8_t* comp_buffer) override
  {
    CommonHeader* common_header = reinterpret_cast<CommonHeader*>(comp_buffer);
    DecompressionConfig decomp_config{status_pool};
    
    gpuErrchk(cudaMemcpy(&decomp_config.decomp_data_size, 
        &common_header->decomp_data_size, 
        sizeof(size_t),
        cudaMemcpyDefault));
    
    do_configure_decompression(decomp_config, common_header);

    return decomp_config;
  }

  void set_scratch_buffer(uint8_t* new_scratch_buffer) final override
  {
    if (scratch_buffer_filled) {
      if (manager_filled_scratch_buffer) {
        #if CUDART_VERSION >= 11020
          gpuErrchk(cudaFreeAsync(scratch_buffer, user_stream));
        #else
          gpuErrchk(cudaFree(scratch_buffer));
        #endif
        manager_filled_scratch_buffer = false;
      }
    } else {
      scratch_buffer_filled = true;
    }
    scratch_buffer = new_scratch_buffer;
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

  /**
   * @brief Required helper that actually does the compression 
   * 
   * @param common_header header filled in by this routine (GPU accessible)
   * @param decomp_buffer The uncompressed input data (GPU accessible)
   * @param decomp_buffer_size The length of the uncompressed input data
   * @param comp_buffer The location to output the compressed data to (GPU accessible).
   * @param comp_config Resulted from configure_compression given this decomp_buffer_size.
   * 
   */
  virtual void do_compress(
      CommonHeader* common_header,
      const uint8_t* decomp_buffer, 
      const size_t decomp_buffer_size, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) = 0;

  /**
   * @brief Required helper that actually does the decompression 
   *
   * @param decomp_buffer The location to output the decompressed data to (GPU accessible).
   * @param comp_buffer The compressed input data (GPU accessible).
   * @param decomp_config Resulted from configure_decompression given this decomp_buffer_size.
   */
  virtual void do_decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config) = 0;

  /**
   * @brief Optionally does additional decompression configuration 
   */
  virtual void do_configure_decompression(
      DecompressionConfig& decomp_config,
      CommonHeader* common_header) = 0; 

  /**
   * @brief Computes the required scratch buffer size 
   */
  virtual size_t compute_scratch_buffer_size() = 0;

  /**
   * @brief Computes the maximum compressed output size for a given
   * uncompressed buffer.
   */
  virtual size_t calculate_max_compressed_output_size(size_t decomp_buffer_size) = 0;

  /**
   * @brief Retrieves a CPU-accessible pointer to the FormatSpecHeader
   */
  virtual FormatSpecHeader* get_format_header() = 0;
};

} // namespace nvcomp