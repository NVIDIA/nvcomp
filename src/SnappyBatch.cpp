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
    size_t /* batch_size */,
    size_t /* max_chunk_size */,
    size_t * temp_bytes)
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
    size_t max_chunk_size,
    size_t * max_compressed_size)
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
	const void* const* device_in_ptr,
	const size_t* device_in_bytes,
	size_t batch_size,
	void* temp_ptr,
	size_t temp_bytes,
	void* const* device_out_ptr,
	size_t* device_out_bytes,
	cudaStream_t stream)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(device_in_ptr);
    CHECK_NOT_NULL(device_in_bytes);
    CHECK_NOT_NULL(device_out_ptr);
    CHECK_NOT_NULL(device_out_bytes);

    size_t * device_out_available_bytes;
    CudaUtils::check(cudaMallocManaged((void **)&device_out_available_bytes,
      sizeof(size_t) * batch_size),
      "Failed to allocate managed memory for device_out_available_bytes of snappy kernel");

    for (size_t i = 0; i < batch_size; i++) {
      device_out_available_bytes[i] = 0;
    }

    gpu_inflate_status_s * statuses;
    CudaUtils::check(cudaMallocManaged((void **)&statuses,
      sizeof(gpu_inflate_status_s) * batch_size),
      "Failed to allocate managed memory for status of snappy kernel");

    CudaUtils::check(cudaStreamAttachMemAsync(stream, device_out_available_bytes),
      "Failed to attach managed memory to stream");
    CudaUtils::check(cudaStreamAttachMemAsync(stream, statuses),
      "Failed to attach managed memory to stream");

    CudaUtils::check(gpu_snap(device_in_ptr, device_in_bytes, device_out_ptr,
        device_out_available_bytes, statuses, device_out_bytes, batch_size, stream),
      "Failed to run gpu_snap");

    CudaUtils::check(cudaStreamSynchronize(stream),
      "Failed to sync on CUDA stream");

    CudaUtils::check(cudaFree(device_out_available_bytes),
      "Failed to free managed memory for device_out_available_bytes of snappy kernel");
    CudaUtils::check(cudaFree(statuses),
      "Failed to free managed memory for status of snappy kernel");

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBatchedSnappyCompressAsync()");
  }

  return nvcompSuccess;
}
