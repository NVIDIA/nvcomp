/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "../Check.h"
#include "../CudaUtils.h"
#include "../common.h"
#include "../type_macros.h"
#include "LZ4Compressor.h"
#include "LZ4Decompressor.h"
#include "LZ4Metadata.h"
#include "LZ4MetadataOnGPU.h"
#include "MutableLZ4MetadataOnGPU.h"
#include "lowlevel/LZ4CompressionKernels.h"

#include "nvcomp.h"
#include "nvcomp.hpp"

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

using namespace nvcomp;
using namespace nvcomp::lowlevel;
using namespace nvcomp::highlevel;

namespace
{

static constexpr size_t DEFAULT_CHUNK_SIZE = 1 << 16;

void check_format_opts(const nvcompLZ4FormatOpts* const format_opts)
{
  CHECK_NOT_NULL(format_opts);

  if (format_opts->chunk_size > lz4MaxChunkSize()) {
    throw std::runtime_error(
        "LZ4 maximum chunk size is " + std::to_string(lz4MaxChunkSize()));
  }
}

size_t get_chunk_size_or_default(const nvcompLZ4FormatOpts* const format_opts)
{
  if (format_opts) {
    check_format_opts(format_opts);
    return format_opts->chunk_size;
  } else {
    return DEFAULT_CHUNK_SIZE;
  }
}

} // namespace

int nvcompLZ4IsMetadata(const void* const metadata_ptr)
{
  const Metadata* const metadata = static_cast<const Metadata*>(metadata_ptr);
  return metadata->getCompressionType() == LZ4Metadata::COMPRESSION_ID;
}

int nvcompLZ4IsData(const void* const in_ptr, size_t in_bytes, cudaStream_t stream)
{
  // Need at least 2 size_t variables to be valid.
  if (in_ptr == NULL || in_bytes < sizeof(size_t)) {
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

void nvcompLZ4DestroyMetadata(void* const metadata_ptr)
{
  delete static_cast<LZ4Metadata*>(metadata_ptr);
}

nvcompStatus_t nvcompLZ4CompressConfigure(
    const nvcompLZ4FormatOpts* const format_opts,
    const nvcompType_t in_type,
    const size_t in_bytes,
    size_t* const metadata_bytes,
    size_t* const temp_bytes,
    size_t* const out_bytes)
{
  try {
    CHECK_NOT_NULL(metadata_bytes);
    CHECK_NOT_NULL(temp_bytes);
    CHECK_NOT_NULL(out_bytes);

    const size_t chunk_bytes = get_chunk_size_or_default(format_opts);

    if (chunk_bytes % sizeOfnvcompType(in_type) != 0) {
      throw std::invalid_argument("Chunk size needs to be a multiple of the input data type");
    }

    *metadata_bytes = sizeof(LZ4Metadata);

    const int total_chunks = roundUpDiv(in_bytes, chunk_bytes);

    *temp_bytes
        = LZ4Compressor::calculate_workspace_size(in_bytes, chunk_bytes);

    const size_t serialized_metadata_bytes
        = LZ4MetadataOnGPU::getSerializedSizeBasedOnChunks(total_chunks);

    const size_t max_comp_bytes
        = LZ4Compressor::calculate_max_output_size(in_bytes, chunk_bytes);

    *out_bytes = serialized_metadata_bytes + max_comp_bytes;
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompLZ4CompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompLZ4CompressAsync(
    const nvcompLZ4FormatOpts* format_opts,
    const nvcompType_t in_type,
    const void* in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(in_ptr);
    CHECK_NOT_NULL(temp_ptr);
    CHECK_NOT_NULL(out_ptr);
    CHECK_NOT_NULL(out_bytes);

    const size_t chunk_bytes = get_chunk_size_or_default(format_opts);

    LZ4Compressor compressor(
        CudaUtils::device_pointer(static_cast<const uint8_t*>(in_ptr)),
        in_bytes,
        chunk_bytes,
        in_type);
    compressor.configure_workspace(
        CudaUtils::device_pointer(temp_ptr), temp_bytes);

    // build the metadatas and configure pointers
    LZ4Metadata metadata(NVCOMP_TYPE_BITS, chunk_bytes, in_bytes, 0);

    MutableLZ4MetadataOnGPU metadataGPU(
        out_ptr, compressor.get_max_output_size());
    metadataGPU.copyToGPU(metadata, stream);

    // the location the prefix sum of the chunks of each item is stored
    size_t* const out_prefix = metadataGPU.compressed_prefix_ptr();

    compressor.configure_output(
        CudaUtils::device_pointer(static_cast<uint8_t*>(out_ptr))
            + metadataGPU.getSerializedSize(),
        out_prefix);
    compressor.compress_async(stream);

    metadataGPU.save_output_size(CudaUtils::device_pointer(out_bytes), stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompLZ4CompressAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompLZ4DecompressConfigure(
    const void* const in_ptr,
    const size_t in_bytes,
    void** const metadata_ptr,
    size_t* const metadata_bytes,
    size_t* const temp_bytes,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  try {
    CHECK_NOT_NULL(metadata_ptr);
    CHECK_NOT_NULL(metadata_bytes);
    CHECK_NOT_NULL(temp_bytes);
    CHECK_NOT_NULL(out_bytes);

    LZ4Metadata* ptr;
    if (in_ptr == nullptr && *metadata_ptr != nullptr) {
      // allow for old API usage, where metadata has already been fetched by a
      // previous call, and we just want to query that structure
      ptr = reinterpret_cast<LZ4Metadata*>(*metadata_ptr);
    } else {
      // fetch metadata
      ptr = new LZ4Metadata(
          LZ4MetadataOnGPU(CudaUtils::device_pointer(in_ptr), in_bytes)
              .copyToHost(stream));
      *metadata_ptr = ptr;
      *metadata_bytes = ptr->getMetadataSize();
    }

    const size_t chunk_size = ptr->getUncompChunkSize();
    const size_t num_chunks = ptr->getNumChunks();

    *temp_bytes
        = LZ4Decompressor::calculate_workspace_size(chunk_size, num_chunks);
    *out_bytes = ptr->getUncompressedSize();
  } catch (std::exception& e) {
    return Check::exception_to_error(e, "nvcompLZ4DecompressConfigure()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompLZ4DecompressAsync(
    const void* const in_ptr,
    const size_t in_bytes,
    const void* const metadata_ptr,
    const size_t /* metadata_bytes */,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    const size_t out_bytes,
    cudaStream_t stream)
{
  try {
    CHECK_NOT_NULL(metadata_ptr);
    CHECK_NOT_NULL(out_ptr);
    CHECK_NOT_NULL(in_ptr);
    CHECK_NOT_NULL(temp_ptr);

    LZ4Metadata* metadata = reinterpret_cast<LZ4Metadata*>((void*)metadata_ptr);

    if (in_bytes < metadata->getCompressedSize()) {
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Input buffer is smaller than compressed data size: "
              + std::to_string(in_bytes) + " < "
              + std::to_string(metadata->getCompressedSize()));
    } else if (out_bytes < metadata->getUncompressedSize()) {
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Output buffer is smaller than the uncompressed data size: "
              + std::to_string(out_bytes) + " < "
              + std::to_string(metadata->getUncompressedSize()));
    }

    const void* const device_in_ptr = CudaUtils::device_pointer(in_ptr);

    LZ4MetadataOnGPU metadataGPU(device_in_ptr, in_bytes);

    const size_t* const comp_prefix = metadataGPU.compressed_prefix_ptr();
    const int num_chunks = metadata->getNumChunks();
    const size_t chunk_size = metadata->getUncompChunkSize();

    LZ4Decompressor decomp(
        static_cast<const uint8_t*>(device_in_ptr)
            + LZ4MetadataOnGPU::getCompressedDataOffset(*metadata),
        comp_prefix,
        in_bytes,
        chunk_size,
        num_chunks);

    decomp.configure_workspace(CudaUtils::device_pointer(temp_ptr), temp_bytes);
    decomp.configure_output(
        CudaUtils::device_pointer(static_cast<uint8_t*>(out_ptr)),
        metadata->getUncompressedSize());
    decomp.decompress_async(stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompLZ4DecompressAsync()");
  }

  return nvcompSuccess;
}
