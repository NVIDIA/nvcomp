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

#ifndef NVCOMP_LZ4_HPP
#define NVCOMP_LZ4_HPP

#include "lz4.h"
#include "nvcomp.hpp"

namespace nvcomp
{
/**
 * @brief C++ wrapper for LZ4 compressor.
 */
class LZ4Compressor : public Compressor
{
public:
  /**
   * @brief Create a new LZ4 compressor.
   *
   * @param chunk_size size of chunks that are eached compressed separately.
   * A value of `0` will result in the default chunk size being used.
   * @param data_type The type of data to compress.
   */
  explicit LZ4Compressor(size_t chunk_size, nvcompType_t data_type);

  /**
   * @brief Create a new LZ4 compressor with the default chunk size.
   */
  LZ4Compressor();

  // disable copying
  LZ4Compressor(const LZ4Compressor&) = delete;
  LZ4Compressor& operator=(const LZ4Compressor&) = delete;

  /**
   * @brief Configure the compressor for the given input, and get the necessary
   * spaces.
   *
   * @param in_bytes The size of the input in bytes.
   * @param temp_bytes The temporary workspace required (output).
   * @param out_bytes The maximum possible output size (output).
   */
  void configure(
      const size_t in_bytes, size_t* temp_bytes, size_t* out_bytes) override;

  /**
   * @brief Perform compression asynchronously.
   *
   * @param in_ptr The uncompressed input data (GPU accessible).
   * @param in_bytes The length of the uncompressed input data.
   * @param temp_ptr The temporary workspace (GPU accessible).
   * @param temp_bytes The size of the temporary workspace.
   * @param out_ptr The location to output data to (GPU accessible).
   * @param out_bytes The size of the output location on input, and the size of
   * the compressed data on output (CPU accessible, but must be pinned or
   * managed memory for this function to be asynchronous).
   * @param stream The stream to operate on.
   *
   * @throw NVCompException If compression fails to launch on the stream.
   */
  void compress_async(
      const void* in_ptr,
      const size_t in_bytes,
      void* temp_ptr,
      const size_t temp_bytes,
      void* out_ptr,
      size_t* out_bytes,
      cudaStream_t stream) override;

private:
  size_t m_chunk_size;
  nvcompType_t m_data_type;
};

class LZ4Decompressor : public Decompressor
{
public:
  LZ4Decompressor();

  ~LZ4Decompressor();

  // disable copying
  LZ4Decompressor(const LZ4Decompressor&) = delete;
  LZ4Decompressor& operator=(const LZ4Decompressor&) = delete;

  /**
   * @brief Configure the decompressor. This synchronizes with the stream.
   *
   * @param in_ptr The compressed data on the device.
   * @param in_bytes The size of the compressed data.
   * @param temp_bytes The temporary space required for decompression (output).
   * @param out_bytes The size of the uncompressed data (output).
   * @param stream The stream to operate on for copying data from the device to
   * the host.
   */
  void configure(
      const void* in_ptr,
      const size_t in_bytes,
      size_t* temp_bytes,
      size_t* out_bytes,
      cudaStream_t stream) override;

  /**
   * @brief Decompress the given data asynchronously.
   *
   * @param temp_ptr The temporary workspace on the device to use.
   * @param temp_bytes The size of the temporary workspace.
   * @param out_ptr The location to write the uncompressed data to on the
   * device.
   * @param out_num_elements The size of the output location in number of
   * elements.
   * @param stream The stream to operate on.
   *
   * @throw NVCompException If decompression fails to launch on the stream.
   */
  void decompress_async(
      const void* in_ptr,
      const size_t in_bytes,
      void* temp_ptr,
      const size_t temp_bytes,
      void* out_ptr,
      const size_t out_bytes,
      cudaStream_t stream) override;

private:
  void* m_metadata_ptr;
  size_t m_metadata_bytes;
};

/******************************************************************************
 * METHOD IMPLEMENTATIONS *****************************************************
 *****************************************************************************/

inline LZ4Compressor::LZ4Compressor(const size_t chunk_size, nvcompType_t data_type) :
    m_chunk_size(chunk_size),
    m_data_type(data_type)
{
  // do nothing
}

inline LZ4Compressor::LZ4Compressor() : LZ4Compressor(0, NVCOMP_TYPE_CHAR)
{
  // do nothing
}

inline void LZ4Compressor::configure(
    const size_t in_bytes, size_t* const temp_bytes, size_t* const out_bytes)
{
  nvcompLZ4FormatOpts opts{m_chunk_size};

  size_t metadata_bytes;
  nvcompStatus_t status = nvcompLZ4CompressConfigure(
      opts.chunk_size == 0 ? nullptr : &opts,
      m_data_type,
      in_bytes,
      &metadata_bytes,
      temp_bytes,
      out_bytes);
  throwExceptionIfError(status, "nvcompLZ4CompressConfigure() failed");
}

inline void LZ4Compressor::compress_async(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  nvcompLZ4FormatOpts opts{m_chunk_size};
  nvcompStatus_t status = nvcompLZ4CompressAsync(
      opts.chunk_size == 0 ? nullptr : &opts,
      m_data_type,
      in_ptr,
      in_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      out_bytes,
      stream);
  throwExceptionIfError(status, "nvcompLZ4CompressAsync() failed");
}

inline LZ4Decompressor::LZ4Decompressor() :
    m_metadata_ptr(nullptr),
    m_metadata_bytes(0)
{
  // do nothing
}

inline LZ4Decompressor::~LZ4Decompressor()
{
  if (m_metadata_ptr) {
    nvcompLZ4DestroyMetadata(m_metadata_ptr);
  }
}

inline void LZ4Decompressor::configure(
    const void* const in_ptr,
    const size_t in_bytes,
    size_t* const temp_bytes,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  nvcompStatus_t status = nvcompLZ4DecompressConfigure(
      in_ptr,
      in_bytes,
      &m_metadata_ptr,
      &m_metadata_bytes,
      temp_bytes,
      out_bytes,
      stream);
  throwExceptionIfError(status, "nvcompLZ4Configure() failed");
}

inline void LZ4Decompressor::decompress_async(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    const size_t out_bytes,
    cudaStream_t stream)
{
  nvcompStatus_t status = nvcompLZ4DecompressAsync(
      in_ptr,
      in_bytes,
      m_metadata_ptr,
      m_metadata_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      out_bytes,
      stream);
  throwExceptionIfError(status, "nvcompLZ4QeueryMetadataAsync() failed");
}

} // namespace nvcomp
#endif
