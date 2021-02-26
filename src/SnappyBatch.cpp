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

#include "nvcomp/snappy.h"

#include "Check.h"
#include "CudaUtils.h"
#include "SnappyKernels.h"
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

//using LZ4MetadataPtr = std::unique_ptr<LZ4Metadata>;

void check_format_opts(const nvcompSnappyFormatOpts* const format_opts)
{
  CHECK_NOT_NULL(format_opts);
}

size_t snappy_get_max_compressed_length(size_t source_bytes) {
  // This is an estimate from the original snappy library 
  return 32 + source_bytes + source_bytes / 6;
}

} // namespace

/******************************************************************************
 *     C-style API calls for BATCHED compression/decompress defined below.
 *****************************************************************************/

nvcompError_t nvcompBatchedSnappyDecompressGetMetadata(
    const void** in_ptr,
    const size_t* in_bytes,
    size_t batch_size,
    void** metadata_ptr,
    cudaStream_t stream)
{
  try {
    throw std::runtime_error("Not implemented");
  } catch (std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedSnappyDecompressGetMetadata()");
  }

  return nvcompSuccess;
}

void nvcompBatchedSnappyDecompressDestroyMetadata(void* metadata_ptr)
{
//  delete static_cast<BatchedLZ4Metadata*>(metadata_ptr);
}

nvcompError_t
nvcompBatchedSnappyDecompressGetTempSize(const void* metadata_ptr, size_t* temp_bytes)
{
  try {
    CHECK_NOT_NULL(metadata_ptr);
    CHECK_NOT_NULL(temp_bytes);

    throw std::runtime_error("Not implemented");
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedSnappyDecompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompError_t
nvcompBatchedSnappyDecompressGetOutputSize(const void* metadata_ptr, size_t batch_size, size_t* output_bytes)
{
  try {
    CHECK_NOT_NULL(metadata_ptr);
    CHECK_NOT_NULL(output_bytes);

    throw std::runtime_error("Not implemented");
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedSnappyDecompressGetOutputSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedSnappyDecompressAsync(
    const void* const* in_ptr,
    const size_t* in_bytes,
    size_t batch_size,
    void* const temp_ptr,
    const size_t temp_bytes,
    const void* metadata_ptr,
    void* const* out_ptr,
    const size_t* out_bytes,
    cudaStream_t stream)
{
  try {
    CHECK_NOT_NULL(metadata_ptr);
    CHECK_NOT_NULL(out_ptr);
    CHECK_NOT_NULL(in_ptr);

    if (temp_bytes > 0) {
      CHECK_NOT_NULL(temp_ptr);
    }

    throw std::runtime_error("Not implemented");
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedSnappyDecompressAsync()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedSnappyCompressGetTempSize(
    const void* const* const /* in_ptr */,
    const size_t* const in_bytes,
    const size_t batch_size,
    const nvcompSnappyFormatOpts* const format_opts,
    size_t* const temp_bytes)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(in_bytes);
    CHECK_NOT_NULL(temp_bytes);
    check_format_opts(format_opts);

    // Snappy doesn't need any workspace in GPU memory
    *temp_bytes = 0;

  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedSnappyCompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedSnappyCompressGetOutputSize(
    const void* const* const /* in_ptr */,
    const size_t* const in_bytes,
    const size_t batch_size,
    const nvcompSnappyFormatOpts* const format_opts,
    void* const /* temp_ptr */,
    const size_t /* temp_bytes */,
    size_t* const out_bytes)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(in_bytes);
    CHECK_NOT_NULL(out_bytes);
    check_format_opts(format_opts);

    for (size_t i = 0; i < batch_size; ++i) {
      out_bytes[i] = snappy_get_max_compressed_length(in_bytes[i]);
    }

  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedSnappyCompressGetOutputSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedSnappyCompressAsync(
    const void* const* const in_ptr,
    const size_t* const in_bytes,
    const size_t batch_size,
    const nvcompSnappyFormatOpts* const format_opts,
    void* const /* temp_ptr */,
    size_t const temp_bytes,
    void* const* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(format_opts);
    CHECK_NOT_NULL(in_ptr);
    CHECK_NOT_NULL(in_bytes);
    CHECK_NOT_NULL(out_ptr);
    CHECK_NOT_NULL(out_bytes);

    gpu_inflate_input_s * compress_inputs;
    CudaUtils::check(cudaMallocManaged((void **)&compress_inputs,
      sizeof(gpu_inflate_input_s) * batch_size),
      "Failed to allocate managed memory for inputs of snappy kernel");

    for (size_t i = 0; i < batch_size; i++) {
      compress_inputs[i].srcDevice = in_ptr[i];
      compress_inputs[i].srcSize = in_bytes[i];
      compress_inputs[i].dstDevice = out_ptr[i];
      // We are assuming we have enough space in the destination buffer
      compress_inputs[i].dstSize = snappy_get_max_compressed_length(in_bytes[i]);
    }

    gpu_inflate_status_s * statuses;
    CudaUtils::check(cudaMallocManaged((void **)&statuses,
      sizeof(gpu_inflate_status_s) * batch_size),
      "Failed to allocate managed memory for status of snappy kernel");

    CudaUtils::check(cudaStreamAttachMemAsync(stream, compress_inputs),
      "Failed to attach managed memory to stream");
    CudaUtils::check(cudaStreamAttachMemAsync(stream, statuses),
      "Failed to attach managed memory to stream");
    CudaUtils::check(gpu_snap(compress_inputs, statuses, batch_size, stream),
      "Failed to run gpu_snap");

    CudaUtils::check(cudaStreamSynchronize(stream),
      "Failed to sync on CUDA stream");

    for (size_t i = 0; i < batch_size; i++)
      out_bytes[i] = statuses[i].bytes_written;

    CudaUtils::check(cudaFree(compress_inputs),
      "Failed to free managed memory for inputs of snappy kernel");
    CudaUtils::check(cudaFree(statuses),
      "Failed to free managed memory for status of snappy kernel");

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedSnappyCompressAsync()");
  }

  return nvcompSuccess;
}
