/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "common.h"
#include "nvcomp.h"
#include "nvcomp/bitcomp.h"
#include "type_macros.h"

#ifdef ENABLE_BITCOMP
#include <bitcomp.h>

nvcompStatus_t nvcompBatchedBitcompCompressGetMaxOutputChunkSize(
    size_t max_chunk_size,
    nvcompBitcompFormatOpts format_opts,
    size_t* max_compressed_size)
{
  *max_compressed_size = bitcompMaxBuflen(max_chunk_size);
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedBitcompCompressAsync(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes, // not used
    size_t batch_size,
    void* device_temp_ptr, // not used
    size_t temp_bytes,     // not used
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    const nvcompBitcompFormatOpts* format_opts,
    nvcompType_t type,
    cudaStream_t stream)
{
    // Convert the NVCOMP type to a BITCOMP type
    bitcompDataType_t dataType;
    switch (type) {
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

    // Create a Bitcomp batch handle, associate it to the stream
    bitcompAlgorithm_t algo = BITCOMP_DEFAULT_ALGO;
    if (format_opts)
        algo = static_cast<bitcompAlgorithm_t>(format_opts->algorithm_type);
    bitcompHandle_t plan;
    if (bitcompCreateBatchPlan(&plan, batch_size, dataType, BITCOMP_LOSSLESS, algo) != BITCOMP_SUCCESS ||
        bitcompSetStream(plan, stream) != BITCOMP_SUCCESS)
    return nvcompErrorInternal;

    // Launch the Bitcomp async batch compression
    if (bitcompBatchCompressLossless(
            plan,
            device_uncompressed_ptrs,
            device_compressed_ptrs,
            device_uncompressed_bytes,
            device_compressed_bytes) != BITCOMP_SUCCESS)
        return nvcompErrorInternal;

    // Once launched, the handle can be destroyed
    if (bitcompDestroyPlan (plan) != BITCOMP_SUCCESS)
        return nvcompErrorInternal;
    
    return nvcompSuccess;
}

// The Bitcomp batch decompression outputs bitcompResult_t statuses.
// Need to convert them to nvcompStatus_t.
__global__ void convertOutputStatuses (nvcompStatus_t *statuses, size_t batch_size)
{
    static_assert (sizeof (nvcompStatus_t) == sizeof (bitcompResult_t));
    size_t index = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (index >= batch_size)
        return;
    bitcompResult_t ier = reinterpret_cast<bitcompResult_t *>(statuses)[index];
    nvcompStatus_t nvcomp_err = nvcompSuccess;
    if (ier != BITCOMP_SUCCESS)
    {
        if (ier == BITCOMP_INVALID_PARAMETER)
            nvcomp_err = nvcompErrorInvalidValue;
        else
            nvcomp_err = nvcompErrorCannotDecompress;
    }
    statuses[index] = nvcomp_err;
}

nvcompStatus_t nvcompBatchedBitcompDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes, // not used
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr, // not used
    size_t temp_bytes,           // not used
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_statuses,
    const nvcompBitcompFormatOpts* format_opts,
    nvcompType_t type,
    cudaStream_t stream)
{
    // Convert the NVCOMP type to a BITCOMP type
    bitcompDataType_t dataType;
    switch (type) {
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

    // Create a Bitcomp batch handle, associate it to the stream
    bitcompAlgorithm_t algo = BITCOMP_DEFAULT_ALGO;
    if (format_opts)
        algo = static_cast<bitcompAlgorithm_t>(format_opts->algorithm_type);
    bitcompHandle_t plan;
    if (bitcompCreateBatchPlan(&plan, batch_size, dataType, BITCOMP_LOSSLESS, algo) != BITCOMP_SUCCESS ||
        bitcompSetStream(plan, stream) != BITCOMP_SUCCESS)
    return nvcompErrorInternal;

    // Launch the Bitcomp async batch decompression with extra checks
    if (bitcompBatchUncompressCheck(
            plan,
            device_compressed_ptrs,
            device_uncompressed_ptrs,
            device_uncompressed_bytes,
            (bitcompResult_t *)device_statuses) != BITCOMP_SUCCESS)
      return nvcompErrorInternal;

    // Need a separate kernel to query the actual uncompressed size,
    // as bitcomp doesn't write the uncompressed size during decompression
    if (bitcompBatchGetUncompressedSizesAsync(
            device_compressed_ptrs,
            device_actual_uncompressed_bytes,
            batch_size,
            stream) != BITCOMP_SUCCESS)
        return nvcompErrorInternal;

    // Also launch a kernel to convert the output statuses
    int blocks = (batch_size + 511) / 512;
    convertOutputStatuses <<<blocks,512,0,stream>>> (device_statuses, batch_size);

    // Once launched, the handle can be destroyed
    if (bitcompDestroyPlan(plan) != BITCOMP_SUCCESS)
        return nvcompErrorInternal;
    return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedBitcompGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream)
{
    if (bitcompBatchGetUncompressedSizesAsync(
            device_compressed_ptrs,
            device_uncompressed_bytes,
            batch_size,
            stream) != BITCOMP_SUCCESS)
        return nvcompErrorInternal;
    return nvcompSuccess;
}

#endif
