/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "../Check.h"
#include "CudaUtils.h"
#include "LZ4Compressor.h"
#include "TempSpaceBroker.h"
#include "common.h"
#include "nvcomp_cub.cuh"

#include "nvcomp/lz4.h"

namespace nvcomp
{
namespace highlevel
{

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

namespace
{

__global__ void copyToContig(
    const uint8_t* const* const chunks,
    const size_t* const chunk_prefix,
    uint8_t* const out)
{
  const size_t chunk = blockIdx.x;

  const size_t offset = chunk_prefix[chunk];
  const size_t size = chunk_prefix[chunk + 1] - offset;

  for (size_t i = threadIdx.x; i < size; i += blockDim.x) {
    out[offset + i] = chunks[chunk][i];
  }
}

__global__ void setupCompressBatchData(
    const uint8_t* const in_ptr,
    const size_t in_size,
    uint8_t* const out_ptr,
    const size_t num_chunks,
    const size_t in_chunk_size,
    const size_t out_chunk_size,
    const void** const in_ptrs,
    size_t* const in_sizes,
    void** const out_ptrs,
    size_t* const output_offset)
{
  const size_t chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;

  assert(in_size <= num_chunks * in_chunk_size);

  if (chunk_idx < num_chunks) {
    in_ptrs[chunk_idx] = in_ptr + in_chunk_size * chunk_idx;
    in_sizes[chunk_idx] = chunk_idx + 1 < num_chunks
                              ? in_chunk_size
                              : ((in_size - 1) % in_chunk_size) + 1;
    out_ptrs[chunk_idx] = out_ptr + out_chunk_size * chunk_idx;
  }

  if (chunk_idx == 0) {
    // zero the first offset
    *output_offset = 0;
  }
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

size_t LZ4Compressor::calculate_workspace_size(
    const size_t decomp_data_size, const size_t chunk_size)
{
  const size_t num_chunks = roundUpDiv(decomp_data_size, chunk_size);

  // needed workspace
  size_t staging_bytes;
  CHECK_API_CALL(nvcompBatchedLZ4CompressGetTempSize(
      num_chunks, chunk_size, nvcompBatchedLZ4DefaultOpts, &staging_bytes));

  // input and output pointers
  const size_t pointer_bytes = 2 * num_chunks * sizeof(uint8_t*);

  // storage for input byte counts, and output bytes counts
  const size_t size_bytes
      = (num_chunks * sizeof(size_t*)) + (num_chunks * sizeof(size_t));

  // buffer to collect output in
  size_t max_out;
  CHECK_API_CALL(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out));
  size_t buffer_bytes = max_out * num_chunks;

  size_t prefix_bytes;
  CudaUtils::check(
      cub::DeviceScan::InclusiveSum(
          nullptr,
          prefix_bytes,
          static_cast<const size_t*>(nullptr),
          static_cast<size_t*>(nullptr),
          num_chunks),
      "cub::DeviceScan::InclusiveSum()");

  return staging_bytes + pointer_bytes + size_bytes + prefix_bytes
         + buffer_bytes;
}

size_t LZ4Compressor::calculate_max_output_size(
    const size_t decomp_data_size, const size_t chunk_size)
{
  const size_t num_chunks = roundUpDiv(decomp_data_size, chunk_size);

  size_t max_out;
  CHECK_API_CALL(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out));

  return max_out * num_chunks;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

LZ4Compressor::LZ4Compressor(
    const uint8_t* decomp_data,
    const size_t decomp_data_size,
    const size_t chunk_size,
    const nvcompType_t data_type) :
    m_input_ptr(decomp_data),
    m_input_size(decomp_data_size),
    m_chunk_size(chunk_size),
    m_data_type(data_type),
    m_num_chunks(roundUpDiv(decomp_data_size, chunk_size)),
    m_output_ptr(nullptr),
    m_output_offsets(nullptr),
    m_workspace(nullptr),
    m_workspace_size(0)
{
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

size_t LZ4Compressor::get_workspace_size() const
{
  return calculate_workspace_size(m_input_size, m_chunk_size);
}

size_t LZ4Compressor::get_max_output_size() const
{
  return calculate_workspace_size(m_input_size, m_chunk_size);
}

void LZ4Compressor::configure_workspace(
    void* const workspace, const size_t size)
{
  const size_t required_Size = get_workspace_size();
  if (size < required_Size) {
    throw std::runtime_error(
        "Insufficient workspace size: " + std::to_string(size) + " / "
        + std::to_string(required_Size));
  } else if (workspace == nullptr) {
    throw std::runtime_error("Workspace cannot be null.");
  }

  m_workspace = workspace;
  m_workspace_size = size;
}

void LZ4Compressor::configure_output(
    uint8_t* const device_location, size_t* const device_offsets)
{
  m_output_ptr = device_location;
  m_output_offsets = device_offsets;
}

void LZ4Compressor::compress_async(cudaStream_t stream)
{
  if (!is_workspace_configured()) {
    throw std::runtime_error(
        "Workspace must be configured before compressing.");
  } else if (!is_output_configured()) {
    throw std::runtime_error("Output must be configured before compressing.");
  }

  TempSpaceBroker temp(m_workspace, m_workspace_size);

  nvcompBatchedLZ4Opts_t opts = { .data_type = m_data_type };

  uint8_t* workspace;
  size_t workspace_size;
  CHECK_API_CALL(nvcompBatchedLZ4CompressGetTempSize(
      m_num_chunks,
      m_chunk_size,
      opts,
      &workspace_size));
  temp.reserve(&workspace, workspace_size);

  size_t max_chunk_output;
  CHECK_API_CALL(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      m_chunk_size, opts, &max_chunk_output));

  // these have all the same size, and generally should on all platforms as
  // the definition of size_t should make it the same size
  static_assert(
      alignof(size_t) == alignof(uint8_t*),
      "Pointers must have the same alignment as size_t");

  const void** in_ptrs_device;
  temp.reserve(&in_ptrs_device, m_num_chunks);

  size_t* in_sizes_device;
  temp.reserve(&in_sizes_device, m_num_chunks);

  void** out_ptrs_device;
  temp.reserve(&out_ptrs_device, m_num_chunks);

  size_t* out_sizes_device;
  temp.reserve(&out_sizes_device, m_num_chunks);

  uint8_t* output_buffer;
  temp.reserve(&output_buffer, max_chunk_output * m_num_chunks);

  {
    const dim3 block(128);
    const dim3 grid(roundUpDiv(m_num_chunks, block.x));

    setupCompressBatchData<<<grid, block, 0, stream>>>(
        m_input_ptr,
        m_input_size,
        output_buffer,
        m_num_chunks,
        m_chunk_size,
        max_chunk_output,
        in_ptrs_device,
        in_sizes_device,
        out_ptrs_device,
        m_output_offsets);
  }

  CHECK_API_CALL(nvcompBatchedLZ4CompressAsync(
      reinterpret_cast<const void* const*>(in_ptrs_device),
      in_sizes_device,
      m_chunk_size,
      m_num_chunks,
      workspace,
      workspace_size,
      reinterpret_cast<void* const*>(out_ptrs_device),
      out_sizes_device,
      opts,
      stream));

  // perform prefixsum on sizes
  size_t prefix_temp_size;
  CudaUtils::check(
      cub::DeviceScan::InclusiveSum(
          nullptr,
          prefix_temp_size,
          out_sizes_device,
          m_output_offsets + 1,
          m_num_chunks,
          stream),
      "cub::DeviceScan::InclusiveSum()");
  void* prefix_temp;
  temp.reserve(&prefix_temp, prefix_temp_size);

  CudaUtils::check(
      cub::DeviceScan::InclusiveSum(
          prefix_temp,
          prefix_temp_size,
          out_sizes_device,
          m_output_offsets + 1,
          m_num_chunks,
          stream),
      "cub::DeviceScan::InclusiveSum()");

  {
    const dim3 grid(m_num_chunks);
    // Since we are copying a whole chunk per thread block, maximize the number
    // of threads we have copying each block
    const dim3 block(1024);

    // Copy prefix sums values to metadata header and copy compressed data into
    // contiguous space
    copyToContig<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t* const*>(out_ptrs_device),
        m_output_offsets,
        m_output_ptr);
    CudaUtils::check_last_error();
  }
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

bool LZ4Compressor::is_workspace_configured() const
{
  return m_workspace != nullptr;
}

bool LZ4Compressor::is_output_configured() const
{
  return m_output_ptr != nullptr;
}

} // namespace highlevel
} // namespace nvcomp
