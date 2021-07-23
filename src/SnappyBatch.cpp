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

size_t snappy_get_max_compressed_length(size_t source_bytes)
{
  // This is an estimate from the original snappy library
  return 32 + source_bytes + source_bytes / 6;
}

} // namespace

/******************************************************************************
 *     C-style API calls for BATCHED compression/decompress defined below.
 *****************************************************************************/

nvcompError_t nvcompBatchedSnappyDecompressGetTempSize(
    size_t /* num_chunks */,
    size_t /* max_uncompressed_chunk_size */,
    size_t* temp_bytes)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(temp_bytes);

    // Snappy doesn't need any workspace in GPU memory
    *temp_bytes = 0;

  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedSnappyDecompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedSnappyGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(device_compressed_ptrs);
    CHECK_NOT_NULL(device_compressed_bytes);
    CHECK_NOT_NULL(device_uncompressed_bytes);

    gpu_get_uncompressed_sizes(
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_bytes,
        batch_size,
        stream);

  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedSnappyGetDecompressSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedSnappyDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const /* temp_ptr */,
    const size_t /* temp_bytes */,
    void* const* device_uncompressed_ptr,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(device_compressed_ptrs);
    CHECK_NOT_NULL(device_compressed_bytes);
    CHECK_NOT_NULL(device_uncompressed_bytes);
    CHECK_NOT_NULL(device_actual_uncompressed_bytes);
    CHECK_NOT_NULL(device_uncompressed_ptr);
    CHECK_NOT_NULL(device_statuses);

    gpu_unsnap(
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_ptr,
        device_uncompressed_bytes,
        (gpu_snappy_status_s*)device_statuses,
        device_actual_uncompressed_bytes,
        batch_size,
        stream);

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedSnappyDecompressAsync()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedSnappyCompressGetTempSize(
    size_t /* batch_size */, size_t /* max_chunk_size */, size_t* temp_bytes)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(temp_bytes);

    // Snappy doesn't need any workspace in GPU memory
    *temp_bytes = 0;

  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedSnappyCompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedSnappyCompressGetOutputSize(
    size_t max_chunk_size, size_t* max_compressed_size)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(max_compressed_size);

    *max_compressed_size = snappy_get_max_compressed_length(max_chunk_size);

  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBatchedSnappyCompressGetOutputSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedSnappyCompressAsync(
    const void* const* device_uncompressed_ptr,
    const size_t* device_uncompressed_bytes,
    size_t /*max_uncompressed_chunk_bytes*/,
    size_t batch_size,
    void* /* device_temp_ptr */,
    size_t /* temp_bytes */,
    void* const* device_compressed_ptr,
    size_t* device_compressed_bytes,
    snappy_opt_type* /*format_ops*/,
    cudaStream_t stream)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(device_uncompressed_ptr);
    CHECK_NOT_NULL(device_uncompressed_bytes);
    CHECK_NOT_NULL(device_compressed_ptr);
    CHECK_NOT_NULL(device_compressed_bytes);

    size_t* device_out_available_bytes = nullptr;
    gpu_snappy_status_s* statuses = nullptr;

    gpu_snap(
        device_uncompressed_ptr,
        device_uncompressed_bytes,
        device_compressed_ptr,
        device_out_available_bytes,
        statuses,
        device_compressed_bytes,
        batch_size,
        stream);

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedSnappyCompressAsync()");
  }

  return nvcompSuccess;
}
