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
#include "src/lowlevel/LZ4CompressionKernels.h"
#include "src/highlevel/LZ4HlifKernels.h"
#include "nvcomp/lz4.h"
#include "src/common.h"
#include "nvcomp_common_deps/hlif_shared_types.h"
#include "BatchManager.hpp"

namespace nvcomp {

struct LZ4FormatSpecHeader {
  nvcompType_t data_type;
};

struct LZ4BatchManager : BatchManager<LZ4FormatSpecHeader> {
private:
  size_t hash_table_size;
  LZ4FormatSpecHeader* format_spec;

public:
  LZ4BatchManager(size_t uncomp_chunk_size, nvcompType_t data_type, cudaStream_t user_stream = 0, const int device_id = 0)
    : BatchManager(uncomp_chunk_size, user_stream, device_id),      
      format_spec()
  {
    gpuErrchk(cudaHostAlloc(&format_spec, sizeof(LZ4FormatSpecHeader), cudaHostAllocDefault));
    format_spec->data_type = data_type;

    max_comp_chunk_size = compute_max_compressed_chunk_size();    
    hash_table_size = lowlevel::lz4GetHashTableSize(max_comp_chunk_size);
    finish_init();
  }

  virtual ~LZ4BatchManager() 
  {
    gpuErrchk(cudaFreeHost(format_spec));
  }

  size_t compute_max_compressed_chunk_size() final override 
  {
    size_t max_comp_chunk_size;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
        uncomp_chunk_size, nvcompBatchedLZ4DefaultOpts, &max_comp_chunk_size);
    return max_comp_chunk_size;
  }

  uint32_t compute_compression_max_block_occupancy() final override 
  {
    return batchedLZ4CompMaxBlockOccupancy(format_spec->data_type, device_id);
  }

  uint32_t compute_decompression_max_block_occupancy() final override 
  {
    return batchedLZ4DecompMaxBlockOccupancy(format_spec->data_type, device_id); 
  }

  LZ4FormatSpecHeader* get_format_header() final override 
  {
    return format_spec;
  }

  void do_batch_compress(
      CommonHeader* common_header,
      const uint8_t* decomp_buffer,
      const size_t decomp_buffer_size,
      uint8_t* comp_data_buffer,
      const uint32_t num_chunks,
      size_t* comp_chunk_offsets,
      size_t* comp_chunk_sizes,
      nvcompStatus_t* output_status) final override
  {
    lz4HlifBatchCompress(
        common_header,
        decomp_buffer,
        decomp_buffer_size,
        comp_data_buffer,
        scratch_buffer,
        uncomp_chunk_size,
        &common_header->comp_data_size,
        ix_chunk,
        num_chunks,
        max_comp_chunk_size,
        hash_table_size,
        comp_chunk_offsets,
        comp_chunk_sizes,
        max_comp_ctas,
        format_spec->data_type,
        user_stream,
        output_status);
  }

  void do_batch_decompress(
      const uint8_t* comp_data_buffer,
      uint8_t* decomp_buffer,
      const uint32_t num_chunks,
      const size_t* comp_chunk_offsets,
      const size_t* comp_chunk_sizes,
      nvcompStatus_t* output_status) final override
  {        
    lz4HlifBatchDecompress(
        comp_data_buffer,
        decomp_buffer,
        uncomp_chunk_size,
        ix_chunk,
        num_chunks,
        comp_chunk_offsets,
        comp_chunk_sizes,
        max_decomp_ctas,
        user_stream,
        output_status);
  }

private: // helper overrides
  size_t compute_scratch_buffer_size() final override
  {
    return max_comp_ctas * (hash_table_size * sizeof(offset_type) 
         + max_comp_chunk_size);
  }  
};

} // namespace nvcomp