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

#include "nvcompManager.hpp"

namespace nvcomp {

template<typename FormatSpecHeader>
struct BatchManager : ManagerBase<FormatSpecHeader> {

protected: // members
  uint32_t* ix_chunk;
  uint32_t max_comp_ctas;
  uint32_t max_decomp_ctas;
  size_t max_comp_chunk_size;
  size_t uncomp_chunk_size;
  using ManagerBase<FormatSpecHeader>::user_stream;

public: // API
  BatchManager(size_t uncomp_chunk_size, cudaStream_t user_stream = 0, int device_id = 0)
    : ManagerBase<FormatSpecHeader>(user_stream, device_id),
      ix_chunk(),
      max_comp_ctas(),
      max_decomp_ctas(),
      max_comp_chunk_size(),
      uncomp_chunk_size(uncomp_chunk_size)
  {
    gpuErrchk(cudaMalloc(&ix_chunk, sizeof(uint32_t)));
  }

  virtual ~BatchManager() {
    gpuErrchk(cudaFree(ix_chunk));
  }

  BatchManager& operator=(const BatchManager&) = delete;     
  BatchManager(const BatchManager&) = delete;     

  void do_decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& config) final override
  {
    const size_t* comp_chunk_offsets = roundUpToAlignment<const size_t>(comp_buffer);
    const size_t* comp_chunk_sizes = comp_chunk_offsets + config.num_chunks;
    const uint32_t* comp_chunk_checksums = reinterpret_cast<const uint32_t*>(comp_chunk_sizes + config.num_chunks);
    const uint32_t* decomp_chunk_checksums = comp_chunk_checksums + config.num_chunks;
    const uint8_t* comp_data_buffer = reinterpret_cast<const uint8_t*>(decomp_chunk_checksums + config.num_chunks);

    gpuErrchk(cudaMemsetAsync(ix_chunk, 0, sizeof(uint32_t), user_stream));
    do_batch_decompress(
        comp_data_buffer,
        decomp_buffer,
        config.num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        config.get_status());
  }

private: // pure virtual functions
  virtual size_t compute_max_compressed_chunk_size() = 0;
  virtual uint32_t compute_compression_max_block_occupancy() = 0;
  virtual uint32_t compute_decompression_max_block_occupancy() = 0;

  virtual void do_batch_compress(
      CommonHeader* common_header,
      const uint8_t* decomp_buffer,
      const size_t decomp_buffer_size,
      uint8_t* comp_data_buffer,
      const uint32_t num_chunks,
      size_t* comp_chunk_offsets,
      size_t* comp_chunk_sizes,
      nvcompStatus_t* output_status) = 0;

  virtual void do_batch_decompress(
      const uint8_t* comp_data_buffer,
      uint8_t* decomp_buffer,
      const uint32_t num_chunks,
      const size_t* comp_chunk_offsets,
      const size_t* comp_chunk_sizes,
      nvcompStatus_t* output_status) = 0;


protected: // derived helpers
  void finish_init() {
    max_comp_ctas = compute_compression_max_block_occupancy();
    max_decomp_ctas = compute_decompression_max_block_occupancy();
    ManagerBase<FormatSpecHeader>::finish_init();
  }

private: // helper API overrides
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

  void do_compress(
      CommonHeader* common_header,
      const uint8_t* decomp_buffer, 
      const size_t decomp_buffer_size, 
      uint8_t* comp_buffer,
      const CompressionConfig& comp_config) final override
  {
    const uint32_t num_chunks = roundUpDiv(decomp_buffer_size, uncomp_chunk_size);
    
    // Pad so that the comp chunk offsets are properly aligned
    size_t* comp_chunk_offsets = roundUpToAlignment<size_t>(comp_buffer);
    size_t* comp_chunk_sizes = comp_chunk_offsets + num_chunks;
    uint32_t* comp_chunk_checksums = reinterpret_cast<uint32_t*>(comp_chunk_sizes + num_chunks);
    uint32_t* decomp_chunk_checksums = comp_chunk_checksums + num_chunks;
    uint8_t* comp_data_buffer = reinterpret_cast<uint8_t*>(decomp_chunk_checksums + num_chunks);

    gpuErrchk(cudaMemsetAsync(ix_chunk, 0, sizeof(uint32_t), user_stream));    
    
    do_batch_compress(
        common_header,
        decomp_buffer,
        decomp_buffer_size,
        comp_data_buffer,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        comp_config.get_status());
  }

  // Can be overridden if the format needs additional scratch space, see LZ4 for an example
  size_t compute_scratch_buffer_size() override
  {
    return max_comp_ctas * max_comp_chunk_size;
  }

};

} // namespace nvcomp