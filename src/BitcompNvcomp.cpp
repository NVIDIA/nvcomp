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

#include "nvcomp/bitcomp.h"
#include "BitcompMetadata.h"

#include "Check.h"
#include "CudaUtils.h"
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

#ifdef ENABLE_BITCOMP

using namespace nvcomp;

nvcompError_t nvcompBitcompDecompressGetMetadata(
    const void* const in_ptr,
    const size_t in_bytes,
    void** const metadata_ptr,
    cudaStream_t stream)
{
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return nvcompErrorCudaError;
  }
  try {
    CHECK_NOT_NULL(metadata_ptr);
    *metadata_ptr = new BitcompMetadata(in_ptr, in_bytes);
  } catch (std::exception& e) {
    return Check::exception_to_error(e, "nvcompBitcompDecompressGetMetadata()");
  }
  return nvcompSuccess;
}

void nvcompBitcompDecompressDestroyMetadata(void* const metadata_ptr)
{
  delete static_cast<BitcompMetadata*>(metadata_ptr);
}

nvcompError_t
nvcompBitcompDecompressGetTempSize(const void* metadata_ptr, size_t* temp_bytes)
{
  // Unused variables kept for consistency. Silence unused warnings
  (void) metadata_ptr;
  *temp_bytes = 0;
  return nvcompSuccess;
}

nvcompError_t nvcompBitcompDecompressGetOutputSize(
    const void* metadata_ptr, size_t* output_bytes)
{
  try {
    CHECK_NOT_NULL(metadata_ptr);
    CHECK_NOT_NULL(output_bytes);
    const BitcompMetadata* metadata
        = static_cast<const BitcompMetadata*>(metadata_ptr);
    *output_bytes = metadata->getUncompressedSize();
  } catch (std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBitcompDecompressGetOutputSize()");
  }
  return nvcompSuccess;
}

nvcompError_t nvcompBitcompDecompressAsync(
    const void* in_ptr,
    size_t in_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    void* metadata_ptr,
    void* out_ptr,
    size_t out_bytes,
    cudaStream_t stream)
{
  // Unused variables kept for consistency. Silence unused warnings
  (void) temp_ptr;
  (void) temp_bytes;
  try {
    CHECK_NOT_NULL(in_ptr);
    CHECK_NOT_NULL(out_ptr);
    CHECK_NOT_NULL(metadata_ptr);
    BitcompMetadata* metadata = static_cast<BitcompMetadata*>(metadata_ptr);
    if (metadata->getCompressedSize() > in_bytes
        || metadata->getUncompressedSize() > out_bytes) {
      throw NVCompException(
          nvcompErrorInvalidValue, "Bitcomp decompression: invalid size(s)");
    }
    bitcompHandle_t handle = metadata->getBitcompHandle();
    if (bitcompSetStream(handle, stream) != BITCOMP_SUCCESS)
      return nvcompErrorInvalidValue;
    if (bitcompUncompress(handle, static_cast<const char*>(in_ptr), out_ptr)
        != BITCOMP_SUCCESS)
      return nvcompErrorInternal;
  } catch (std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompBitcompDecompressGetOutputSize()");
  }
  return nvcompSuccess;
}

nvcompError_t nvcompBitcompCompressGetTempSize(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    size_t* temp_bytes)
{
  // Unused variables kept for consistency. Silence unused warnings
  (void) in_ptr;
  (void) in_bytes;
  (void) in_type;
  *temp_bytes = 0;
  return nvcompSuccess;
}

nvcompError_t nvcompBitcompCompressGetOutputSize(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    const nvcompBitcompFormatOpts* format_opts,
    void* temp_ptr,
    size_t temp_bytes,
    size_t* out_bytes,
    int exact_out_bytes)
{
  // Unused variables kept for consistency. Silence unused warnings
  (void) in_ptr;
  (void) in_type;
  (void) format_opts;
  (void) temp_ptr;
  (void) temp_bytes;  
  if (exact_out_bytes)
    return nvcompErrorNotSupported;
  *out_bytes = bitcompMaxBuflen(in_bytes);
  return nvcompSuccess;
}

nvcompError_t nvcompBitcompCompressAsync(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    const nvcompBitcompFormatOpts* format_opts,
    void* temp_ptr,
    size_t temp_bytes,
    void* out_ptr,
    size_t* out_bytes,
    cudaStream_t stream)
{
  // Unused variables kept for consistency. Silence unused warnings
  (void) temp_ptr;
  (void) temp_bytes;
  bitcompDataType_t dataType;
  switch (in_type) {
  case NVCOMP_TYPE_CHAR:
    dataType = BITCOMP_SIGNED_8BIT;
    break;
  case NVCOMP_TYPE_USHORT:
    dataType = BITCOMP_UNSIGNED_16BIT;
    break;
  case NVCOMP_TYPE_SHORT:
    dataType = BITCOMP_SIGNED_16BIT;
    break;
  case NVCOMP_TYPE_UINT:
    dataType = BITCOMP_UNSIGNED_32BIT;
    break;
  case NVCOMP_TYPE_INT:
    dataType = BITCOMP_SIGNED_32BIT;
    break;
  case NVCOMP_TYPE_ULONGLONG:
    dataType = BITCOMP_UNSIGNED_64BIT;
    break;
  case NVCOMP_TYPE_LONGLONG:
    dataType = BITCOMP_SIGNED_64BIT;
    break;
  default:
    dataType = BITCOMP_UNSIGNED_8BIT;
  }
  bitcompAlgorithm_t algo = format_opts->algorithm_type ? BITCOMP_SPARSE_ALGO
                                                        : BITCOMP_DEFAULT_ALGO;
  bitcompHandle_t handle;
  bitcompResult_t ier;
  ier = bitcompCreatePlan(&handle, in_bytes, dataType, BITCOMP_LOSSLESS, algo);
  if (ier != BITCOMP_SUCCESS)
    return nvcompErrorInternal;
  if (bitcompSetStream(handle, stream) != BITCOMP_SUCCESS)
    return nvcompErrorInvalidValue;
  if (bitcompCompressLossless(handle, in_ptr, static_cast<char*>(out_ptr))
      != BITCOMP_SUCCESS)
    return nvcompErrorInternal;
  if (bitcompDestroyPlan(handle) != BITCOMP_SUCCESS)
    return nvcompErrorInternal;
  // Not really async since we need to return the compressed size...
  if (cudaStreamSynchronize(stream) != cudaSuccess)
    return nvcompErrorInternal;
  if (bitcompGetCompressedSize (static_cast<char*>(out_ptr), out_bytes) != BITCOMP_SUCCESS)
    return nvcompErrorInternal;
  return nvcompSuccess;
}

int nvcompIsBitcompData(const void* const in_ptr, size_t in_bytes)
{
  bitcompResult_t ier;
  size_t compressedSize, uncompressedSize;
  bitcompDataType_t dataType;
  bitcompMode_t mode;
  bitcompAlgorithm_t algo;
  compressedSize = in_bytes;
  ier = bitcompGetCompressedInfo(
      static_cast<const char*>(in_ptr),
      &compressedSize,
      &uncompressedSize,
      &dataType,
      &mode,
      &algo);
  if (ier == BITCOMP_SUCCESS)
    return 1;
  return 0;
}

#endif // ENABLE_BITCOMP