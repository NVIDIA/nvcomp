/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "lz4.h"

#include "CudaUtils.h"
#include "LZ4BatchCompressor.h"
#include "LZ4CompressionKernels.h"
#include "LZ4Metadata.h"
#include "LZ4MetadataOnGPU.h"
#include "MutableBatchedLZ4MetadataOnGPU.h"
#include "common.h"
#include "nvcomp.h"
#include "nvcomp.hpp"
#include "type_macros.h"

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>


using namespace nvcomp;

/******************************************************************************
 *     C-style API calls for BATCHED compression/decompress defined below.
 *****************************************************************************/

nvcompError_t nvcompBatchedLZ4DecompressGetMetadata(
    const void** in_ptr,
    size_t* in_bytes,
    size_t batch_size,
    void** metadata_ptr,
    cudaStream_t stream)
{
  std::vector<LZ4Metadata*> batch_metadata;
  batch_metadata.reserve(batch_size);

  try {

    for(size_t i=0; i<batch_size; i++) {
      nvcompError_t err;
      LZ4Metadata* m;

      err = nvcompLZ4DecompressGetMetadata(
              in_ptr[i], in_bytes[i], (void**)&m, stream);
      if(err != nvcompSuccess) {
        throw err;
      }
      batch_metadata.emplace_back(std::move(m));
    }

    cudaStreamSynchronize(stream);

    *metadata_ptr = new std::vector<LZ4Metadata*>(std::move(batch_metadata));
  }
  catch (nvcompError_t err) {
    for(size_t i=0; i<batch_metadata.size(); i++) {
      nvcompLZ4DecompressDestroyMetadata(batch_metadata[i]);
    }
    return err;
  }

  return nvcompSuccess;
}

void nvcompBatchedLZ4DecompressDestroyMetadata(void* metadata_ptr)
{
  std::vector<LZ4Metadata*>& metadata = *static_cast<std::vector<LZ4Metadata*>*>(metadata_ptr);

  for(size_t i=0; i<metadata.size(); i++) {
    nvcompLZ4DecompressDestroyMetadata(metadata[i]);
  }
}

nvcompError_t
nvcompBatchedLZ4DecompressGetTempSize(const void* metadata_ptr, size_t* temp_bytes)
{

  if (temp_bytes == NULL) {
    std::cerr << "Invalid, temp_bytes ptr NULL." << std::endl;
    return nvcompErrorInvalidValue;
  }

  if (metadata_ptr == NULL) {
    std::cerr << "Invalid, metadata ptr NULL." << std::endl;
    return nvcompErrorInvalidValue;
  }

  try {
    std::vector<LZ4Metadata*>& metadata
        = *static_cast<std::vector<LZ4Metadata*>*>((void*)metadata_ptr);

    const size_t batch_size = metadata.size();

    size_t total_temp_bytes=0;
    for(size_t b=0;  b<batch_size; b++) {
      const size_t chunk_size = metadata[b]->getUncompChunkSize();

      const size_t num_chunks = metadata[b]->getNumChunks();

      size_t this_temp_bytes
          = lz4DecompressComputeTempSize(num_chunks, chunk_size);

      total_temp_bytes += this_temp_bytes;
    }
    *temp_bytes = total_temp_bytes;
    
  } catch (const std::exception& e) {
    std::cerr << "Failed to get temp size for batch: " << e.what() << std::endl;
    return nvcompErrorCudaError;
  }

  return nvcompSuccess;
}

nvcompError_t
nvcompBatchedLZ4DecompressGetOutputSize(const void* metadata_ptr, size_t batch_size, size_t* output_bytes)
{
  if (metadata_ptr == NULL) {
    std::cerr << "Invalid, metadata NULL." << std::endl;
    return nvcompErrorInvalidValue;
  }

  std::vector<LZ4Metadata*>& metadata = *static_cast<std::vector<LZ4Metadata*>*>((void*)metadata_ptr);

  for(size_t i=0; i<batch_size; i++) {
    nvcompError_t err;
    err = nvcompLZ4DecompressGetOutputSize(metadata[i], &output_bytes[i]);

    if(err != nvcompSuccess) {
      return err;
    }
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedLZ4DecompressAsync(
    const void* const* in_ptr,
    const size_t* in_bytes,
    size_t batch_size,
    void* const temp_ptr,
    const size_t temp_bytes,
    const void* metadata_ptr,
    void* const* out_ptr,
    const size_t* /*out_bytes*/,
    cudaStream_t stream)
{

  if (metadata_ptr == NULL) {
    std::cerr << "Invalid, metadata NULL." << std::endl;
    return nvcompErrorInvalidValue;
  }

  std::vector<LZ4Metadata*>& metadata = *static_cast<std::vector<LZ4Metadata*>*>((void*)metadata_ptr);
  metadata.reserve(batch_size);
  std::vector<const size_t*> comp_prefix;
  comp_prefix.reserve(batch_size);
  std::vector<int> chunks_in_item;
  chunks_in_item.reserve(batch_size);

  for (size_t i = 1; i < batch_size; ++i) {
    if (metadata[i]->getUncompChunkSize() != metadata[i-1]->getUncompChunkSize()) {
      std::cerr << "Cannot decompress items in the same batch with different chunk sizes." << std::endl;
      return nvcompErrorNotSupported;
    }
  }

  for(size_t i=0; i<batch_size; i++) {
    if (in_bytes[i] < metadata[i]->getCompressedSize()) {
      std::cerr << "Input buffer of input " << i 
                << " is smaller than compressed data size: "
                << in_bytes[i] << " < " << metadata[i]->getCompressedSize()
                << std::endl;
      return nvcompErrorInvalidValue;
    }

    LZ4MetadataOnGPU metadataGPU(in_ptr[i], in_bytes[i]);

    comp_prefix.emplace_back(metadataGPU.compressed_prefix_ptr());
    chunks_in_item.emplace_back(metadata[i]->getNumChunks());
  }

  lz4DecompressBatches(
      temp_ptr,
      temp_bytes,
      out_ptr,
      reinterpret_cast<const uint8_t* const*>(in_ptr),
      batch_size,
      comp_prefix.data(),
      metadata[0]->getUncompChunkSize(), // All batches have some chunk size
      chunks_in_item.data(),
      stream);

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedLZ4CompressGetTempSize(
    const void* const* const /* in_ptr */,
    const size_t* const in_bytes,
    const size_t batch_size,
    const nvcompLZ4FormatOpts* const format_opts,
    size_t* const temp_bytes)
{
  try {
    if (in_bytes == nullptr) {
      throw std::runtime_error("in_bytes must not be null.");
    } else if (format_opts == nullptr) {
      throw std::runtime_error("format_opts must not be null.");
    } else if (temp_bytes == nullptr) {
      throw std::runtime_error("temp_bytes must not be null.");
    }

    if (format_opts->chunk_size < lz4MinChunkSize()) {
      throw std::runtime_error(
          "LZ4 minimum chunk size is " + std::to_string(lz4MinChunkSize()));
    }

    *temp_bytes = LZ4BatchCompressor::calculate_workspace_size(
        in_bytes, batch_size, format_opts->chunk_size);

  } catch (const std::exception& e) {
    std::cerr << "Failed to get temp size for batch: " << e.what() << std::endl;
    return nvcompErrorCudaError;
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedLZ4CompressGetOutputSize(
    const void* const* const in_ptr,
    const size_t* const in_bytes,
    const size_t batch_size,
    const nvcompLZ4FormatOpts* const format_opts,
    void* const /* temp_ptr */,
    const size_t /* temp_bytes */,
    size_t* const out_bytes)
{
  try {
    // error check inputs
    if (format_opts == nullptr) {
      throw std::runtime_error("format_opts must not be null.");
    } else if (in_ptr == nullptr) {
      throw std::runtime_error("in_ptr must not be null.");
    } else if (in_bytes == nullptr) {
      throw std::runtime_error("in_bytes must not be null.");
    } else if (out_bytes == nullptr) {
      throw std::runtime_error("out_bytes must not be null.");
    }

    if (format_opts->chunk_size < lz4MinChunkSize()) {
      throw std::runtime_error(
          "LZ4 minimum chunk size is " + std::to_string(lz4MinChunkSize()));
    }

    for (size_t b = 0; b < batch_size; ++b) {
      if (in_ptr[b] == nullptr) {
        throw std::runtime_error(
            "in_ptr[" + std::to_string(b) + "] must not be null.");
      }

      const size_t chunk_bytes = format_opts->chunk_size;
      const int total_chunks = roundUpDiv(in_bytes[b], chunk_bytes);

      const size_t metadata_bytes
          = LZ4Metadata::OffsetAddr * sizeof(size_t)
            + ((total_chunks + 1)
               * sizeof(size_t)); // 1 extra val to store total length

      const size_t max_comp_bytes
          = lz4ComputeMaxSize(chunk_bytes) * total_chunks;

      out_bytes[b] = metadata_bytes + max_comp_bytes;
    }

  } catch (const std::exception& e) {
    std::cerr << "Failed to get output size for batch: " << e.what()
              << std::endl;
    return nvcompErrorCudaError;
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedLZ4CompressAsync(
    const void* const* const in_ptr,
    const size_t* const in_bytes,
    const size_t batch_size,
    const nvcompLZ4FormatOpts* const format_opts,
    void* const temp_ptr,
    size_t const temp_bytes,
    void* const* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  try {
    // error check inputs
    if (format_opts == nullptr) {
      throw std::runtime_error("format_opts must not be null.");
    } else if (in_ptr == nullptr) {
      throw std::runtime_error("in_ptr must not be null.");
    } else if (in_bytes == nullptr) {
      throw std::runtime_error("in_bytes must not be null.");
    } else if (temp_ptr == nullptr) {
      throw std::runtime_error("temp_ptr must not be null.");
    } else if (out_ptr == nullptr) {
      throw std::runtime_error("out_ptr must not be null.");
    } else if (out_bytes == nullptr) {
      throw std::runtime_error("out_bytes must not be null.");
    }

    const size_t chunk_bytes = format_opts->chunk_size;

    // build the metadatas and configure pointers
    std::vector<LZ4Metadata> metadata;
    metadata.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      metadata.emplace_back(NVCOMP_TYPE_BITS, chunk_bytes, in_bytes[i], 0);
    }

    MutableBatchedLZ4MetadataOnGPU metadataGPU(out_ptr, out_bytes, batch_size);

    std::vector<size_t> out_data_start(batch_size);
    metadataGPU.copyToGPU(
        metadata, temp_ptr, temp_bytes, out_data_start.data(), stream);

    const uint8_t* const* const typed_in_ptr
        = reinterpret_cast<const uint8_t* const*>(in_ptr);
    LZ4BatchCompressor compressor(
        typed_in_ptr, in_bytes, batch_size, chunk_bytes);

    compressor.configure_workspace(temp_ptr, temp_bytes);

    // the location the prefix sum of the chunks of each item is stored
    std::vector<size_t*> out_prefix(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      out_prefix[i] = metadataGPU.compressed_prefix_ptr(i);
    }

    uint8_t* const* const typed_out_ptr
        = reinterpret_cast<uint8_t* const*>(out_ptr);
    compressor.configure_output(
        typed_out_ptr, out_prefix.data(), out_data_start.data(), out_bytes);

    compressor.compress_async(stream);

    return nvcompSuccess;
  } catch (const std::exception& e) {
    std::cerr << "Failed launch compression for batch: " << e.what()
              << std::endl;
    return nvcompErrorCudaError;
  }

  return nvcompSuccess;
}
