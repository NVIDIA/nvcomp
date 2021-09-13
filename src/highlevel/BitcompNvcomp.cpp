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

#include "BitcompMetadata.h"
#include "nvcomp/bitcomp.h"

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
using namespace nvcomp::highlevel;

void nvcompBitcompDestroyMetadata(void* const metadata_ptr)
{
  delete static_cast<BitcompMetadata*>(metadata_ptr);
}

nvcompStatus_t nvcompBitcompDecompressConfigure(
    const void* compressed_ptr,
    size_t compressed_bytes,
    void** metadata_ptr,
    size_t* metadata_bytes,
    size_t* temp_bytes,
    size_t* uncompressed_bytes,
    cudaStream_t stream)
{
  try {
    CHECK_NOT_NULL(metadata_ptr);

    // as Bitcomp pulls the metadata from the default stream, sync the
    // current stream first.
    CudaUtils::sync(stream);
    *metadata_ptr = new BitcompMetadata(compressed_ptr, compressed_bytes);
    *metadata_bytes = sizeof(BitcompMetadata);

    *temp_bytes = 0;
    *uncompressed_bytes = reinterpret_cast<BitcompMetadata*>(*metadata_ptr)
                              ->getUncompressedSize();
  } catch (std::exception& e) {
    return Check::exception_to_error(e, "nvcompBitcompDecompressConfigure()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompBitcompDecompressAsync(
    const void* const in_ptr,
    size_t in_bytes,
    void* const metadata_ptr,
    const size_t /* metadata_bytes */,
    void* const /* temp_ptr */,
    const size_t /* temp_bytes */,
    void* const out_ptr,
    size_t out_bytes,
    cudaStream_t stream)
{
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
    return Check::exception_to_error(e, "nvcompBitcompDecompressAsync()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompBitcompCompressConfigure(
    const nvcompBitcompFormatOpts* const /* opts */,
    const nvcompType_t /* in_type */,
    const size_t in_bytes,
    size_t* const metadata_bytes,
    size_t* const temp_bytes,
    size_t* const max_compressed_bytes)
{
  try {
    CHECK_NOT_NULL(metadata_bytes);
    CHECK_NOT_NULL(temp_bytes);
    CHECK_NOT_NULL(max_compressed_bytes);

    *metadata_bytes = sizeof(BitcompMetadata);
    *temp_bytes = 0;
    *max_compressed_bytes = bitcompMaxBuflen(in_bytes);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompBitcompCompressConfigure()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBitcompCompressAsync(
    const nvcompBitcompFormatOpts* format_opts,
    nvcompType_t in_type,
    const void* in_ptr,
    size_t in_bytes,
    void* /* temp_ptr */,
    size_t /* temp_bytes */,
    void* out_ptr,
    size_t* out_bytes,
    cudaStream_t stream)
{
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
  bitcompAlgorithm_t algo = BITCOMP_DEFAULT_ALGO;
  if (format_opts) {
    algo = static_cast<bitcompAlgorithm_t>(format_opts->algorithm_type);
  }

  bitcompHandle_t handle;
  bitcompResult_t ier;
  ier = bitcompCreatePlan(&handle, in_bytes, dataType, BITCOMP_LOSSLESS, algo);
  if (ier != BITCOMP_SUCCESS)
    return nvcompErrorInternal;
  if (bitcompSetStream(handle, stream) != BITCOMP_SUCCESS)
    return nvcompErrorInvalidValue;
  if (bitcompCompressLossless(handle, in_ptr, out_ptr) != BITCOMP_SUCCESS)
    return nvcompErrorInternal;
  if (bitcompDestroyPlan(handle) != BITCOMP_SUCCESS)
    return nvcompErrorInternal;
  if (bitcompGetCompressedSizeAsync(out_ptr, out_bytes, stream) != BITCOMP_SUCCESS)
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
      in_ptr,
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
