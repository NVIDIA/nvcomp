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
#include "LZ4Decompressor.h"
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

__global__ void setupDecompressBatchData(
    const uint8_t* const in_ptr,
    uint8_t* const out_ptr,
    const size_t* const in_prefix,
    const size_t num_chunks,
    const size_t chunk_size,
    const size_t out_size,
    const void** const in_ptrs,
    size_t* const in_sizes,
    void** const out_ptrs,
    size_t* const out_sizes)
{
  const size_t chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;

  assert(out_size <= num_chunks * chunk_size);

  if (chunk_idx < num_chunks) {
    in_ptrs[chunk_idx] = in_ptr + in_prefix[chunk_idx];
    in_sizes[chunk_idx] = in_prefix[chunk_idx + 1] - in_prefix[chunk_idx];
    out_ptrs[chunk_idx] = out_ptr + chunk_size * chunk_idx;
    out_sizes[chunk_idx] = chunk_idx + 1 < num_chunks
                               ? chunk_size
                               : ((out_size - 1) % chunk_size) + 1;
  }
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

size_t LZ4Decompressor::calculate_workspace_size(
    const size_t chunk_size, const size_t num_chunks)
{
  // compute tempspace
  size_t c_api_space;
  CHECK_API_CALL(nvcompBatchedLZ4DecompressGetTempSize(
      num_chunks, chunk_size, &c_api_space));

  // we need an input and output pointer for each chunk
  const size_t pointer_bytes = sizeof(void*) * num_chunks * 2;
  // we need input and output sizes for each chunk
  const size_t size_bytes = sizeof(size_t) * num_chunks * 2;
  // we need actual output sizes
  const size_t actual_size_bytes = sizeof(size_t) * num_chunks;
  // we need the error status for each chunk
  const size_t status_bytes = sizeof(nvcompStatus_t) * num_chunks;

  return c_api_space + pointer_bytes + size_bytes + actual_size_bytes
         + status_bytes;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

LZ4Decompressor::LZ4Decompressor(
    const uint8_t* const comp_data,
    const size_t* const comp_prefix,
    const size_t comp_data_size,
    const size_t chunk_size,
    const size_t num_chunks) :
    m_input_size(comp_data_size),
    m_chunk_size(chunk_size),
    m_num_chunks(num_chunks),
    m_input_ptr(comp_data),
    m_input_prefix(comp_prefix),
    m_output_ptr(nullptr),
    m_output_size(0),
    m_workspace(nullptr),
    m_workspace_size(0)
{
  // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

size_t LZ4Decompressor::get_workspace_size() const
{
  return calculate_workspace_size(m_chunk_size, m_num_chunks);
}

void LZ4Decompressor::configure_workspace(
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

void LZ4Decompressor::configure_output(
    uint8_t* const device_location, const size_t output_size)
{
  m_output_ptr = device_location;
  m_output_size = output_size;
}

void LZ4Decompressor::decompress_async(cudaStream_t stream)
{
  if (!is_workspace_configured()) {
    throw std::runtime_error(
        "Workspace must be configured before compressing.");
  } else if (!is_output_configured()) {
    throw std::runtime_error("Output must be configured before compressing.");
  }

  TempSpaceBroker temp(m_workspace, m_workspace_size);

  uint8_t* workspace;
  size_t workspace_size;
  CHECK_API_CALL(nvcompBatchedLZ4DecompressGetTempSize(
      m_num_chunks, m_chunk_size, &workspace_size));
  temp.reserve(&workspace, workspace_size);

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

  size_t* actual_out_sizes_devices;
  temp.reserve(&actual_out_sizes_devices, m_num_chunks);

  nvcompStatus_t* device_statuses;
  temp.reserve(&device_statuses, m_num_chunks);

  const dim3 block(128);
  const dim3 grid(roundUpDiv(m_num_chunks, block.x));

  setupDecompressBatchData<<<grid, block, 0, stream>>>(
      m_input_ptr,
      m_output_ptr,
      m_input_prefix,
      m_num_chunks,
      m_chunk_size,
      m_output_size,
      in_ptrs_device,
      in_sizes_device,
      out_ptrs_device,
      out_sizes_device);

  CHECK_API_CALL(nvcompBatchedLZ4DecompressAsync(
      in_ptrs_device,
      in_sizes_device,
      out_sizes_device,
      actual_out_sizes_devices,
      m_num_chunks,
      workspace,
      workspace_size,
      out_ptrs_device,
      device_statuses,
      stream));
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

bool LZ4Decompressor::is_workspace_configured() const
{
  return m_workspace != nullptr;
}

bool LZ4Decompressor::is_output_configured() const
{
  return m_output_ptr != nullptr;
}

} // namespace highlevel
} // namespace nvcomp
