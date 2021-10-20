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

#ifndef NVCOMP_BITCOMP_HPP
#define NVCOMP_BITCOMP_HPP

#include <type_traits>

#include "nvcomp.hpp"
#include "nvcomp/bitcomp.h"

#ifdef ENABLE_BITCOMP

namespace nvcomp
{

/**
 * @brief C++ wrapper for Bitcomp compressor.
 */
class BitcompCompressor : public Compressor
{
public:
  /**
   * @brief Create a new Bitcomp compressor.
   *
   * @param type The data type being compressed.
   * @param algorithm_type The type of algorithm to use for compression.
   *    0 : Default algorithm, usually gives the best compression ratios
   *    1 : "Sparse" algorithm, works well on sparse data (with lots of zeroes).
   *        and is usually a faster than the default algorithm.
   */
  BitcompCompressor(nvcompType_t type, int algorithm_type);

  /**
   * @brief Create a new Bitcomp compressor with the default algorithm.
   *
   * @param type The data type being compressed.
   */
  explicit BitcompCompressor(nvcompType_t type);

  // disable copying
  BitcompCompressor(const BitcompCompressor&) = delete;
  BitcompCompressor& operator=(const BitcompCompressor&) = delete;

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
   * the compressed data on output (GPU accessible).
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
  nvcompType_t m_type;
  int m_algorithm_type;
};

class BitcompDecompressor : public Decompressor
{
public:
  BitcompDecompressor();

  ~BitcompDecompressor();

  // disable copying
  BitcompDecompressor(const BitcompDecompressor&) = delete;
  BitcompDecompressor& operator=(const BitcompDecompressor&) = delete;

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

inline BitcompCompressor::BitcompCompressor(
    const nvcompType_t type, const int algorithm_type) :
    m_type(type),
    m_algorithm_type(algorithm_type)
{
  // do nothing
}

inline BitcompCompressor::BitcompCompressor(const nvcompType_t type) :
    BitcompCompressor(type, -1)
{
  // do nothing
}

inline void BitcompCompressor::configure(
    const size_t in_bytes, size_t* const temp_bytes, size_t* const out_bytes)
{
  nvcompBitcompFormatOpts opts{m_algorithm_type};

  size_t metadata_bytes;
  nvcompStatus_t status = nvcompBitcompCompressConfigure(
      opts.algorithm_type == -1 ? nullptr : &opts,
      m_type,
      in_bytes,
      &metadata_bytes,
      temp_bytes,
      out_bytes);
  throwExceptionIfError(status, "nvcompBitcompCompressConfigure() failed");
}

inline void BitcompCompressor::compress_async(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  nvcompBitcompFormatOpts opts{m_algorithm_type};
  nvcompStatus_t status = nvcompBitcompCompressAsync(
      opts.algorithm_type == -1 ? nullptr : &opts,
      m_type,
      in_ptr,
      in_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      out_bytes,
      stream);
  throwExceptionIfError(status, "nvcompBitcompCompressAsync() failed");
}

inline BitcompDecompressor::BitcompDecompressor() :
    m_metadata_ptr(nullptr),
    m_metadata_bytes(0)
{
  // do nothing
}

inline BitcompDecompressor::~BitcompDecompressor()
{
  if (m_metadata_ptr) {
    nvcompBitcompDestroyMetadata(m_metadata_ptr);
  }
}

inline void BitcompDecompressor::configure(
    const void* const in_ptr,
    const size_t in_bytes,
    size_t* const temp_bytes,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  nvcompStatus_t status = nvcompBitcompDecompressConfigure(
      in_ptr,
      in_bytes,
      &m_metadata_ptr,
      &m_metadata_bytes,
      temp_bytes,
      out_bytes,
      stream);
  throwExceptionIfError(status, "nvcompBitcompConfigure() failed");
}

inline void BitcompDecompressor::decompress_async(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    const size_t out_bytes,
    cudaStream_t stream)
{
  nvcompStatus_t status = nvcompBitcompDecompressAsync(
      in_ptr,
      in_bytes,
      m_metadata_ptr,
      m_metadata_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      out_bytes,
      stream);
  throwExceptionIfError(status, "nvcompBitcompQeueryMetadataAsync() failed");
}

} // namespace nvcomp

#endif // ENABLE_BITCOMP

#endif
