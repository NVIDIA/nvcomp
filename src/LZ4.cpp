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

#include "LZ4CompressionKernels.h"
#include "LZ4Metadata.h"
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

namespace
{
constexpr const size_t CHUNKS_PER_BATCH = 2048;
}

/**************************************************************************************
 *                New C-style API calls defined below.
 * They use many of the functions defined farther below, but provide an
 *easier-to-use wrapper around them that follows the new API outlined in the
 *google doc.
 *****************************)*********************************************************/

int LZ4IsData(const void* const in_ptr, size_t in_bytes)
{
  // Need at least 2 size_t variables to be valid.
  if(in_ptr == NULL || in_bytes < sizeof(size_t)) {
    return false;
  }
  size_t header_val;
  cudaMemcpy(&header_val, in_ptr, sizeof(size_t), cudaMemcpyDeviceToHost);
  return (header_val == LZ4_FLAG); 
}

int LZ4IsMetadata(const void* const metadata_ptr)
{
  const Metadata* const metadata = static_cast<const Metadata*>(metadata_ptr);
  return metadata->getCompressionType() == LZ4Metadata::COMPRESSION_ID;
}

nvcompError_t nvcompLZ4DecompressGetMetadata(
    const void* const in_ptr,
    const size_t in_bytes,
    void** const metadata_ptr,
    cudaStream_t stream)
{
  try {
    size_t metadata_bytes;
    cudaError_t err;

    // Get size of metadata object
    err = cudaMemcpyAsync(
        &metadata_bytes,
        ((size_t*)in_ptr) + 1,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        stream);
    if(err != cudaSuccess) {
      throw std::runtime_error(
          "Failed to launch copy of metadata bytes "
          "size from device to host."
          + std::to_string(err));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "Failed to sync after copy of metadata "
          "bytes size from device to host: "
          + std::to_string(err));
    }

    if (in_bytes < metadata_bytes) {
      throw std::runtime_error(
          "Compressed data is too small to contain "
          "metadata of size "
          + std::to_string(metadata_bytes) + " / " + std::to_string(in_bytes));
    }

    std::vector<char> metadata_buffer(metadata_bytes);
    err = cudaMemcpyAsync(
        metadata_buffer.data(),
        in_ptr,
        metadata_bytes,
        cudaMemcpyDeviceToHost,
        stream);
    if(err != cudaSuccess) {
      throw std::runtime_error(
          "Failed to launch copy metadata from device "
          " to host: "
          + std::to_string(err));
    }

    *metadata_ptr
        = new LZ4Metadata(metadata_buffer.data(), metadata_buffer.size());

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "Failed to sync after copy of metadata "
          "from device to host: "
          + std::to_string(err));
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception in nvcompLZ4DecompressGetMetadata: " << e.what()
              << std::endl;
    return nvcompErrorInvalidValue;
  }

  return nvcompSuccess;
}

void nvcompLZ4DecompressDestroyMetadata(void* const metadata_ptr)
{
  LZ4Metadata* metadata = static_cast<LZ4Metadata*>(metadata_ptr);
  ::operator delete(metadata);
}

nvcompError_t
nvcompLZ4DecompressGetTempSize(const void* metadata_ptr, size_t* temp_bytes)
{
  if (temp_bytes == NULL) {
    std::cerr << "Invalid, temp_bytes ptr NULL." << std::endl;
    return nvcompErrorInvalidValue;
  }
  // LZ4 decompression doesn't need any temp memory
  *temp_bytes = 0;

  if (metadata_ptr == NULL) {
    std::cerr << "Invalid, metadata ptr NULL." << std::endl;
    return nvcompErrorInvalidValue;
  }
  return nvcompSuccess;
}

nvcompError_t
nvcompLZ4DecompressGetOutputSize(const void* metadata_ptr, size_t* output_bytes)
{
  *output_bytes = static_cast<const LZ4Metadata*>(metadata_ptr)
                      ->getUncompressedSize();

  return nvcompSuccess;
}

nvcompError_t nvcompLZ4DecompressAsync(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const /*temp_ptr*/,
    const size_t /*temp_bytes*/,
    const void* const metadata_ptr,
    void* const out_ptr,
    const size_t /*out_bytes*/,
    cudaStream_t stream)
{

  if (metadata_ptr == NULL) {
    std::cerr << "Invalid, metadata NULL." << std::endl;
    return nvcompErrorInvalidValue;
  }

  const LZ4Metadata* const metadata
      = static_cast<const LZ4Metadata*>(metadata_ptr);

  if (in_bytes < metadata->getCompressedSize()) {
    std::cerr << "Input buffer is smaller than compressed data size: "
              << in_bytes << " < " << metadata->getCompressedSize()
              << std::endl;
    return nvcompErrorInvalidValue;
  }

  lz4DecompressBatch(
      out_ptr,
      in_ptr,
      4 * sizeof(size_t), // Header has metadata size, decomp_size, and
                          // chunk_size before offsets
      metadata->getUncompChunkSize(),
      metadata->getUncompressedSize() % metadata->getUncompChunkSize(),
      metadata->getNumChunks(),
      stream);

  return nvcompSuccess;
}

nvcompError_t nvcompLZ4CompressGetTempSize(
    const void* /*in_ptr*/,
    const size_t in_bytes,
    nvcompType_t /*in_type*/,
    const nvcompLZ4FormatOpts* const format_opts,
    size_t* const temp_bytes)
{
  try {
    if (format_opts == nullptr) {
      throw std::runtime_error("Format opts must not be null.");
    }

    size_t batch_size = format_opts->chunk_size * CHUNKS_PER_BATCH;
    if (in_bytes < batch_size) {
      batch_size = in_bytes;
    }

    const size_t num_chunks = roundUpDiv(in_bytes, format_opts->chunk_size);
    const size_t req_temp_size = lz4ComputeTempSize(
        std::min(CHUNKS_PER_BATCH, num_chunks), format_opts->chunk_size);

    *temp_bytes = req_temp_size;
  } catch (const std::exception& e) {
    std::cerr << "Failed to get temp size: " << e.what() << std::endl;
    return nvcompErrorCudaError;
  }

  return nvcompSuccess;
}

nvcompError_t nvcompLZ4CompressGetOutputSize(
    const void* /*in_ptr*/,
    const size_t in_bytes,
    const nvcompType_t /*in_type*/,
    const nvcompLZ4FormatOpts* format_opts,
    void* const /*temp_ptr*/,
    const size_t /*temp_bytes*/,
    size_t* const out_bytes,
    const int exact_out_bytes)
{

  try 
  {
    if (exact_out_bytes) {
      throw std::runtime_error("Exact output bytes is unimplemented at "
                               "this time.");
    }

    const size_t chunk_bytes = format_opts->chunk_size;
    const int total_chunks = roundUpDiv(in_bytes, chunk_bytes);

    const size_t metadata_bytes
        = 4 * sizeof(size_t)
          + (total_chunks + 1)
                * sizeof(size_t); // 1 extra val to store total length

    const size_t max_comp_bytes = lz4ComputeMaxSize(in_bytes);

    *out_bytes = metadata_bytes + max_comp_bytes;
  } catch (const std::exception& e) {
    std::cerr << "LZ4CompressGetOutputSize() Exception: " << e.what()
              << std::endl;
    return nvcompErrorInvalidValue;
  }
  return nvcompSuccess;
}

nvcompError_t nvcompLZ4CompressAsync(
    const void* in_ptr,
    const size_t in_bytes,
    const nvcompType_t /*in_type*/,
    const nvcompLZ4FormatOpts* format_opts,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  const size_t chunk_bytes = format_opts->chunk_size;

  const int total_chunks = roundUpDiv(in_bytes, chunk_bytes);
  const int batch_chunks = CHUNKS_PER_BATCH;

  const size_t metadata_bytes
      = 4 * sizeof(size_t)
        + (total_chunks + 1)
              * sizeof(size_t); // 1 extra val to store total length

  const size_t start_offset
      = metadata_bytes; // Offset starts after metadata headers

  const size_t LZ4_flag = LZ4_FLAG;
  cudaMemcpyAsync((void*)out_ptr, &LZ4_flag, sizeof(size_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(
      (void*)(((size_t*)out_ptr) + LZ4Metadata::MetadataBytes),
      &metadata_bytes,
      sizeof(size_t),
      cudaMemcpyHostToDevice,
      stream);
  cudaMemcpyAsync(
      (void*)(((size_t*)out_ptr) + LZ4Metadata::UncompressedSize),
      &in_bytes,
      sizeof(size_t),
      cudaMemcpyHostToDevice,
      stream);
  cudaMemcpyAsync(
      (void*)(((size_t*)out_ptr) + LZ4Metadata::ChunkSize),
      &chunk_bytes,
      sizeof(size_t),
      cudaMemcpyHostToDevice,
      stream);
  cudaMemcpyAsync(
      (void*)(((size_t*)out_ptr) + LZ4Metadata::OffsetAddr),
      &start_offset,
      sizeof(size_t),
      cudaMemcpyHostToDevice,
      stream);

  const uint8_t* in_progress = static_cast<const uint8_t*>(in_ptr);
  uint8_t* metadata_progress = (uint8_t*)out_ptr + (5*sizeof(size_t)); // Offsets stored after initial header info

  size_t batch_bytes = batch_chunks * chunk_bytes;
  for(size_t batch_offset=0; batch_offset < in_bytes; batch_offset += batch_bytes) {
    if (in_bytes < batch_bytes + batch_offset) {
      batch_bytes = in_bytes - batch_offset;
    }
    const int chunks = roundUpDiv(batch_bytes, chunk_bytes);

    lz4CompressBatch(
        ((uint8_t*)out_ptr),
        temp_ptr,
        temp_bytes,
        in_progress,
        metadata_progress,
        batch_bytes,
        (size_t)std::min(chunk_bytes, batch_bytes),
        chunks,
        chunks,
        stream);

    // Advance pointers
    in_progress += batch_bytes;
    metadata_progress = metadata_progress+(chunks*sizeof(size_t));
  }

// Copy the ending offset of the last chunk (final prefix sums value) out as the total compressed size
  cudaMemcpyAsync(out_bytes, &(((uint8_t*)out_ptr)[4*sizeof(size_t)+total_chunks*sizeof(size_t)]), sizeof(size_t), cudaMemcpyDeviceToHost, stream);

  return nvcompSuccess;
}
