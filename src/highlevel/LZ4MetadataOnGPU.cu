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

#include "Check.h"
#include "CudaUtils.h"
#include "LZ4MetadataOnGPU.h"

#include <cassert>
#include <stdexcept>

namespace nvcomp
{
namespace highlevel
{

/******************************************************************************
 * CUDA KERNELS ***************************************************************
 *****************************************************************************/

namespace
{

__global__ void computeTotalSize(
    const size_t metadata_size,
    const size_t* const compressed_data_size,
    size_t* const output_size)
{
  *output_size = metadata_size + *compressed_data_size;
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

size_t LZ4MetadataOnGPU::getSerializedSizeOf(const LZ4Metadata& metadata)
{
  return getSerializedSizeBasedOnChunks(metadata.getNumChunks());
}

size_t LZ4MetadataOnGPU::getSerializedSizeBasedOnChunks(const size_t num_chunks)
{
  return (LZ4Metadata::OffsetAddr + num_chunks + 1) * sizeof(size_t);
}

size_t LZ4MetadataOnGPU::getCompressedDataOffset(const LZ4Metadata& metadata)
{
  return getSerializedSizeOf(metadata);
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

LZ4MetadataOnGPU::LZ4MetadataOnGPU(
    const void* const ptr, const size_t maxSize) :
    m_ptr(ptr),
    m_max_size(maxSize),
    m_num_chunks(0),
    m_serialized_size(0)
{
  if (ptr == nullptr) {
    throw std::runtime_error("Cannot have nullptr for metadata location.");
  }
}

LZ4MetadataOnGPU::LZ4MetadataOnGPU(const LZ4MetadataOnGPU& other) :
    LZ4MetadataOnGPU(other.m_ptr, other.m_max_size)
{
  m_num_chunks = other.m_num_chunks;
  m_serialized_size = other.m_serialized_size;
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

LZ4MetadataOnGPU& LZ4MetadataOnGPU::operator=(const LZ4MetadataOnGPU& other)
{
  m_ptr = other.m_ptr;
  m_max_size = other.m_max_size;
  m_num_chunks = other.m_num_chunks;
  m_serialized_size = other.m_serialized_size;

  return *this;
}

size_t LZ4MetadataOnGPU::getSerializedSize() const
{
  if (m_serialized_size == 0) {
    throw std::runtime_error("Serialized size has not been set.");
  }
  assert(m_num_chunks > 0);
  CHECK_EQ(
      m_serialized_size,
      (LZ4Metadata::OffsetAddr + m_num_chunks + 1) * sizeof(size_t));

  return m_serialized_size;
}

const size_t* LZ4MetadataOnGPU::compressed_prefix_ptr() const
{
  return static_cast<const size_t*>(m_ptr) + LZ4Metadata::OffsetAddr;
}

LZ4Metadata LZ4MetadataOnGPU::copyToHost(cudaStream_t stream)
{
  size_t metadata_bytes;
  CudaUtils::copy_async(
      &metadata_bytes,
      ((size_t*)m_ptr) + LZ4Metadata::MetadataBytes,
      1,
      DEVICE_TO_HOST,
      stream);
  CudaUtils::sync(stream);

  if (metadata_bytes > m_max_size) {
    throw std::runtime_error(
        "Compressed data is too small to contain "
        "metadata of size "
        + std::to_string(metadata_bytes) + " / " + std::to_string(m_max_size));
  }

  std::vector<uint8_t> metadata_buffer(metadata_bytes);
  CudaUtils::copy_async(
      metadata_buffer.data(),
      static_cast<const uint8_t*>(m_ptr),
      metadata_bytes,
      DEVICE_TO_HOST,
      stream);
  CudaUtils::sync(stream);

  set_serialized_size(metadata_bytes);

  LZ4Metadata metadata
      = LZ4Metadata(metadata_buffer.data(), metadata_buffer.size());
  assert(getSerializedSizeOf(metadata) == metadata_bytes);

  return metadata;
}

void LZ4MetadataOnGPU::save_output_size(
    size_t* const device_size, cudaStream_t stream) const
{
  computeTotalSize<<<1, 1, 0, stream>>>(
      getSerializedSize(), compressed_prefix_ptr() + m_num_chunks, device_size);
}

/******************************************************************************
 * PROTECTED METHODS **********************************************************
 *****************************************************************************/

size_t LZ4MetadataOnGPU::max_size() const
{
  return m_max_size;
}

void LZ4MetadataOnGPU::set_serialized_size(const size_t size)
{
  m_serialized_size = size;
}

void LZ4MetadataOnGPU::set_num_chunks(const size_t chunks)
{
  m_num_chunks = chunks;
}

} // namespace highlevel
} // namespace nvcomp
