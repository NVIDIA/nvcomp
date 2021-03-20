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

#include "nvcomp/lz4.h"

#include "BatchedLZ4Metadata.h"
#include "Check.h"
#include "CudaUtils.h"
#include "LZ4BatchCompressor.h"
#include "LZ4Metadata.h"
#include "LZ4MetadataOnGPU.h"
#include "MutableBatchedLZ4MetadataOnGPU.h"
#include "MutableLZ4MetadataOnGPU.h"
#include "common.h"
#include "lowlevel/LZ4CompressionKernels.h"
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
using namespace nvcomp::lowlevel;

using LZ4MetadataPtr = std::unique_ptr<LZ4Metadata>;

namespace
{

void check_format_opts(const nvcompLZ4FormatOpts* const format_opts)
{
  CHECK_NOT_NULL(format_opts);

  if (format_opts->chunk_size < lz4MinChunkSize()) {
    throw std::runtime_error(
        "LZ4 minimum chunk size is " + std::to_string(lz4MinChunkSize()));
  } else if (format_opts->chunk_size > lz4MaxChunkSize()) {
    throw std::runtime_error(
        "LZ4 maximum chunk size is " + std::to_string(lz4MaxChunkSize()));
  }
}

LZ4MetadataPtr get_individual_metadata(
    const void* const in_ptr, const size_t in_bytes, cudaStream_t stream)
{
  // Get size of metadata object
  size_t metadata_bytes;
  CudaUtils::copy_async(
      &metadata_bytes, ((const size_t*)in_ptr) + 1, 1, DEVICE_TO_HOST, stream);
  CudaUtils::sync(stream);

  if (in_bytes < metadata_bytes) {
    throw std::runtime_error(
        "Compressed data is too small to contain "
        "metadata of size "
        + std::to_string(metadata_bytes) + " / " + std::to_string(in_bytes));
  }

  std::vector<char> metadata_buffer(metadata_bytes);
  CudaUtils::copy_async(
      metadata_buffer.data(),
      (const char*)in_ptr,
      metadata_bytes,
      DEVICE_TO_HOST,
      stream);
  CudaUtils::sync(stream);

  return LZ4MetadataPtr(
      new LZ4Metadata(metadata_buffer.data(), metadata_buffer.size()));
}

} // namespace

int LZ4IsMetadata(const void* const metadata_ptr)
{
  const Metadata* const metadata = static_cast<const Metadata*>(metadata_ptr);
  return metadata->getCompressionType() == LZ4Metadata::COMPRESSION_ID;
}

int LZ4IsData(const void* const in_ptr, size_t in_bytes, cudaStream_t stream)
{
  // Need at least 2 size_t variables to be valid.
  if(in_ptr == NULL || in_bytes < sizeof(size_t)) {
    return false;
  }
  size_t header_val;
  CudaUtils::copy_async(
      &header_val,
      static_cast<const size_t*>(in_ptr),
      1,
      DEVICE_TO_HOST,
      stream);
  CudaUtils::sync(stream);
  return (header_val == LZ4_FLAG); 
}

nvcompError_t nvcompLZ4DecompressGetMetadata(
    const void* const in_ptr,
    const size_t in_bytes,
    void** const metadata_ptr,
    cudaStream_t stream)
{
  try {
    BatchedLZ4Metadata batch_metadata;

    LZ4MetadataPtr m = get_individual_metadata(in_ptr, in_bytes, stream);
    batch_metadata.add(std::move(m));

    cudaStreamSynchronize(stream);

    *metadata_ptr = new BatchedLZ4Metadata(std::move(batch_metadata));
  } catch (std::exception& e) {
    return Check::exception_to_error(e, "nvcompLZ4DecompressGetMetadata()");
  }

  return nvcompSuccess;
}

void nvcompLZ4DecompressDestroyMetadata(void* const metadata_ptr)
{
  delete static_cast<BatchedLZ4Metadata*>(metadata_ptr);
}

nvcompError_t
nvcompLZ4DecompressGetTempSize(const void* metadata_ptr, size_t* temp_bytes)
{
  try {
    CHECK_NOT_NULL(metadata_ptr);
    CHECK_NOT_NULL(temp_bytes);

    const BatchedLZ4Metadata& metadata
        = *static_cast<const BatchedLZ4Metadata*>((void*)metadata_ptr);

    const size_t batch_size = metadata.size();

    size_t total_temp_bytes = 0;
    for (size_t b = 0; b < batch_size; b++) {
      const size_t chunk_size = metadata[b]->getUncompChunkSize();

      const size_t num_chunks = metadata[b]->getNumChunks();

      size_t this_temp_bytes
          = lz4DecompressComputeTempSize(num_chunks, chunk_size);

      total_temp_bytes += this_temp_bytes;
    }
    *temp_bytes = total_temp_bytes;

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompDecompressGetTempSize()");
  }
  return nvcompSuccess;
}

nvcompError_t
nvcompLZ4DecompressGetOutputSize(const void* metadata_ptr, size_t* output_bytes)
{
  const size_t batch_size = 1;

  try {
    CHECK_NOT_NULL(metadata_ptr);
    CHECK_NOT_NULL(output_bytes);

    BatchedLZ4Metadata& metadata
        = *static_cast<BatchedLZ4Metadata*>((void*)metadata_ptr);

    CHECK_EQ(batch_size, metadata.size());

    for (size_t i = 0; i < batch_size; i++) {
      output_bytes[i] = metadata[i]->getUncompressedSize();
    }
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedLZ4DecompressGetOutputSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompLZ4DecompressAsync(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    const void* const metadata_ptr,
    void* const out_ptr,
    const size_t out_bytes,
    cudaStream_t stream)
{
  try {
    CHECK_NOT_NULL(metadata_ptr);
    CHECK_NOT_NULL(out_ptr);
    CHECK_NOT_NULL(in_ptr);

    if (temp_bytes > 0) {
      CHECK_NOT_NULL(temp_ptr);
    }

    BatchedLZ4Metadata& metadata
        = *static_cast<BatchedLZ4Metadata*>((void*)metadata_ptr);

    if (in_bytes < metadata[0]->getCompressedSize()) {
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Input buffer is smaller than compressed data size: "
              + std::to_string(in_bytes) + " < "
              + std::to_string(metadata[0]->getCompressedSize()));
    } else if (out_bytes < metadata[0]->getUncompressedSize()) {
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Output buffer is smaller than the uncompressed data size: "
              + std::to_string(out_bytes) + " < "
              + std::to_string(metadata[0]->getUncompressedSize()));
    }

    LZ4MetadataOnGPU metadataGPU(in_ptr, in_bytes);

    const size_t* const comp_prefix = metadataGPU.compressed_prefix_ptr();
    const int chunks_in_item = metadata[0]->getNumChunks();

    lz4DecompressBatches(
        temp_ptr,
        temp_bytes,
        &out_ptr,
        reinterpret_cast<const uint8_t* const*>(&in_ptr),
        1,
        &comp_prefix,
        metadata[0]->getUncompChunkSize(),
        &chunks_in_item,
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedLZ4DecompressAsync()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompLZ4CompressGetTempSize(
    const void* const /* in_ptr */,
    const size_t in_bytes,
    nvcompType_t /*in_type*/,
    const nvcompLZ4FormatOpts* const format_opts,
    size_t* const temp_bytes)
{
  const size_t batch_size = 1;

  try {
    CHECK_NOT_NULL(temp_bytes);
    check_format_opts(format_opts);

    *temp_bytes = LZ4BatchCompressor::calculate_workspace_size(
        &in_bytes, batch_size, format_opts->chunk_size);

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompLZ4CompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompLZ4CompressGetOutputSize(
    const void* const in_ptr,
    const size_t in_bytes,
    const nvcompType_t /*in_type*/,
    const nvcompLZ4FormatOpts* format_opts,
    void* const /* temp_ptr */,
    const size_t /* temp_bytes */,
    size_t* const out_bytes,
    const int exact_out_bytes)
{
  if (exact_out_bytes) {
    std::cerr
        << "LZ4CompressGetOutputSize(): Exact output bytes is unimplemented at "
           "this time."
        << std::endl;
    return nvcompErrorInvalidValue;
  }

  try {
    // error check inputs
    CHECK_NOT_NULL(in_ptr);
    CHECK_NOT_NULL(out_bytes);
    check_format_opts(format_opts);

    if (in_ptr == nullptr) {
      throw std::runtime_error("in_ptr must not be null.");
    }

    const size_t chunk_bytes = format_opts->chunk_size;
    const int total_chunks = roundUpDiv(in_bytes, chunk_bytes);

    const size_t metadata_bytes
        = LZ4Metadata::OffsetAddr * sizeof(size_t)
          + ((total_chunks + 1)
             * sizeof(size_t)); // 1 extra val to store total length

    const size_t max_comp_bytes = lz4ComputeMaxSize(chunk_bytes) * total_chunks;

    *out_bytes = metadata_bytes + max_comp_bytes;
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompLZ4CompressGetOutputSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompLZ4CompressAsync(
    const void* in_ptr,
    const size_t in_bytes,
    const nvcompType_t in_type,
    const nvcompLZ4FormatOpts* format_opts,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(format_opts);
    CHECK_NOT_NULL(in_ptr);
    CHECK_NOT_NULL(temp_ptr);
    CHECK_NOT_NULL(out_ptr);
    CHECK_NOT_NULL(out_bytes);

    // check if we have enough room to output
    size_t req_out_bytes;
    CHECK_API_CALL(nvcompLZ4CompressGetOutputSize(
        in_ptr,
        in_bytes,
        in_type,
        format_opts,
        temp_ptr,
        temp_bytes,
        &req_out_bytes,
        0));
    if (req_out_bytes > *out_bytes) {
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Output size is too small: " + std::to_string(*out_bytes) + " / "
              + std::to_string(req_out_bytes)
              + " required bytes. Make sure "
                "to set the size of out_bytes to size of output space "
                "allocated.");
    }

    const size_t chunk_bytes = format_opts->chunk_size;

    // build the metadatas and configure pointers
    std::vector<LZ4Metadata> metadata{
        {NVCOMP_TYPE_BITS, chunk_bytes, in_bytes, 0}};

    MutableBatchedLZ4MetadataOnGPU metadataGPU(&out_ptr, out_bytes, 1);

    size_t out_data_start = 0;
    metadataGPU.copyToGPU(
        metadata, temp_ptr, temp_bytes, &out_data_start, stream);

    const uint8_t* const* const typed_in_ptr
        = reinterpret_cast<const uint8_t* const*>(&in_ptr);
    LZ4BatchCompressor compressor(typed_in_ptr, &in_bytes, 1, chunk_bytes);

    compressor.configure_workspace(temp_ptr, temp_bytes);

    // the location the prefix sum of the chunks of each item is stored
    size_t* const out_prefix = metadataGPU.compressed_prefix_ptr(0);

    uint8_t* const* const typed_out_ptr
        = reinterpret_cast<uint8_t* const*>(&out_ptr);
    compressor.configure_output(
        typed_out_ptr, &out_prefix, &out_data_start, out_bytes);

    compressor.compress_async(stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompLZ4CompressAsync()");
  }

  return nvcompSuccess;
}
