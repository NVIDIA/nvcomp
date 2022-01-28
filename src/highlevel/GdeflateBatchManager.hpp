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
#include <memory>

#include "src/Check.h"
#include "src/CudaUtils.h"
#include "nvcomp/gdeflate.h"
#include "src/common.h"
#include "nvcomp_common_deps/hlif_shared_types.h"
#include "BatchManager.hpp"

#include "GdeflateHlifKernels.h"

namespace nvcomp {

struct GdeflateBatchManager : BatchManager<nvcompBatchedGdeflateOpts_t> {
private:
  size_t hash_table_size;
  nvcompBatchedGdeflateOpts_t* format_spec;

public:
  GdeflateBatchManager(size_t uncomp_chunk_size, int algo, cudaStream_t user_stream = 0, const int device_id = 0)
    : BatchManager(uncomp_chunk_size, user_stream, device_id),      
      hash_table_size(),
      format_spec()
  {
    switch(algo) {
      case (0) :
        break;
      case(1) :
        throw std::invalid_argument("Invalid format_opts.algo value (high compression option (1) not currently supported)");
        break;
      default :
        throw std::invalid_argument("Invalid format_opts.algo value (not 0 or 1)");
    }

    CudaUtils::check(cudaHostAlloc(&format_spec, sizeof(nvcompBatchedGdeflateOpts_t), cudaHostAllocDefault));
    format_spec->algo = algo;

    finish_init();
  }

  virtual ~GdeflateBatchManager() 
  {
    CudaUtils::check(cudaFreeHost(format_spec));
  }

  GdeflateBatchManager(const GdeflateBatchManager&) = delete;
  GdeflateBatchManager& operator=(const GdeflateBatchManager&) = delete;

  size_t compute_max_compressed_chunk_size() final override 
  {
    size_t max_comp_chunk_size;
    nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
        get_uncomp_chunk_size(), *format_spec, &max_comp_chunk_size);
    return max_comp_chunk_size;
  }

  uint32_t compute_compression_max_block_occupancy() final override 
  {
    return gdeflate::hlif::batchedGdeflateCompMaxBlockOccupancy(device_id);
  }

  uint32_t compute_decompression_max_block_occupancy() final override 
  {
    return gdeflate::hlif::batchedGdeflateDecompMaxBlockOccupancy(device_id); 
  }

  nvcompBatchedGdeflateOpts_t* get_format_header() final override 
  {
    return format_spec;
  }

  void do_batch_compress(const CompressArgs& compress_args) final override
  {
    gdeflate::hlif::gdeflateHlifBatchCompress(
        compress_args,
        get_max_comp_ctas(),
        user_stream);
  }

  void do_batch_decompress(
      const uint8_t* comp_data_buffer,
      uint8_t* decomp_buffer,
      const uint32_t num_chunks,
      const size_t* comp_chunk_offsets,
      const size_t* comp_chunk_sizes,
      nvcompStatus_t* output_status) final override
  {        
    gdeflate::hlif::gdeflateHlifBatchDecompress(
        comp_data_buffer,
        decomp_buffer,
        get_uncomp_chunk_size(),
        ix_chunk,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        get_max_decomp_ctas(),
        user_stream,
        output_status);
  }

private: // helper overrides
  size_t compute_scratch_buffer_size() final override
  {
    // TODO: reuse this code from gdeflate
    constexpr size_t gdeflate_hash_table_size = 1U << 14;
    size_t chunk_size = get_uncomp_chunk_size();
    size_t tmp_space = sizeof(unsigned int)          + // num_symbols
                       sizeof(unsigned int)          + // num_literals
                       chunk_size * sizeof(uint16_t) + // length
                       chunk_size * sizeof(uint16_t) + // distance
                       chunk_size * sizeof(uint8_t)  + // literals
                       gdeflate_hash_table_size * sizeof(uint16_t); // Hash tables
    tmp_space = ((tmp_space + 3) & (~3)); // round up to nearest 4 byte
    return get_max_comp_ctas() * (tmp_space + get_max_comp_chunk_size());
  }  

  void format_specific_init() final override {}
};

} // namespace nvcomp
