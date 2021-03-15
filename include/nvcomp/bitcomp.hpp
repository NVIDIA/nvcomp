/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <bitcomp.h>

namespace nvcomp
{

/**
 * @brief Bitcomp compressor offered by nvcomp
 *
 * @tparam T The type to compress.
 */
template <typename T>
class BitcompCompressor : public Compressor<T>
{

public:
  /**
   * @brief Create a new BitcompCompressor.
   *
   * @param in_ptr The input data on the GPU to compress.
   * @param num_elements The number of elements to compress.
   * @param algorithm The bitcomp algorithm selector (0 = default, 1=sparse)
   */
  BitcompCompressor(
      const T* in_ptr, const size_t num_elements, int algorithm = 0) :
      Compressor<T>(in_ptr, num_elements), m_algorithm(algorithm)
  {
  }

  // disable copying
  BitcompCompressor(const BitcompCompressor&) = delete;
  BitcompCompressor& operator=(const BitcompCompressor&) = delete;

  /**
   * @brief Get size of the temporary workspace in bytes, required to perform
   * compression.
   *
   * @return The size in bytes.
   */
  size_t get_temp_size() override
  {
    size_t temp_bytes;
    nvcompError_t status = nvcompBitcompCompressGetTempSize(
        this->get_uncompressed_data(),
        this->get_uncompressed_size(),
        this->get_type(),
        &temp_bytes);
    throwExceptionIfError(status, "get_temp_size failed");
    return temp_bytes;
  }

  /**
   * @brief Get the exact size the data will compress to. This can be used in
   * place of `get_max_output_size()` to get the minimum size of the
   * allocation that should be passed to `compress()`. This however, may take
   * similar amount of time to compression itself, and may execute synchronously
   * on the device.
   *
   * For Cascaded compression, this is not yet implemented, and will always
   * throw an exception.
   *
   * @param comp_temp The temporary workspace.
   * @param comp_temp_bytes The size of the temporary workspace.
   *
   * @return The exact size in bytes.
   *
   * @throw NVCompressionException will always be thrown.
   */
  size_t get_exact_output_size(void* comp_temp, size_t comp_temp_bytes) override
  {
    nvcompBitcompFormatOpts opts;
    opts.algorithm_type = m_algorithm;
    size_t out_bytes;
    nvcompError_t status = nvcompBitcompCompressGetOutputSize(
        this->get_uncompressed_data(),
        this->get_uncompressed_size(),
        this->get_type(),
        &opts,
        comp_temp,
        comp_temp_bytes,
        &out_bytes,
        1);
    throwExceptionIfError(status, "get_exact_output_size failed");
    return out_bytes;
  }

  /**
   * @brief Get the maximum size the data could compressed to. This is the
   * upper bound of the minimum size of the allocation that should be
   * passed to `compress()`.
   *
   * @param comp_temp The temporary workspace.
   * @param comp_temp_bytes THe size of the temporary workspace.
   *
   * @return The maximum size in bytes.
   */
  size_t get_max_output_size(void* comp_temp, size_t comp_temp_bytes) override
  {
    nvcompBitcompFormatOpts opts;
    opts.algorithm_type = m_algorithm;
    size_t out_bytes;
    nvcompError_t status = nvcompBitcompCompressGetOutputSize(
        this->get_uncompressed_data(),
        this->get_uncompressed_size(),
        this->get_type(),
        &opts,
        comp_temp,
        comp_temp_bytes,
        &out_bytes,
        0);
    throwExceptionIfError(status, "get_max_output_size failed");
    return out_bytes;
  }

private:
  /**
   * @brief Perform compression asynchronously.
   *
   * @param temp_ptr The temporary workspace on the device.
   * @param temp_bytes The size of the temporary workspace.
   * @param out_ptr The output location the the device (for compressed data).
   * @param out_bytes The size of the output location on the device on input,
   * and the size of the compressed data on output.
   * @param stream The stream to operate on.
   *
   * @throw NVCompException If compression fails to launch on the stream.
   */
  void do_compress(
      void* temp_ptr,
      size_t temp_bytes,
      void* out_ptr,
      size_t* out_bytes,
      cudaStream_t stream) override
  {
    nvcompBitcompFormatOpts opts;
    opts.algorithm_type = m_algorithm;
    nvcompError_t status = nvcompBitcompCompressAsync(
        this->get_uncompressed_data(),
        this->get_uncompressed_size(),
        this->get_type(),
        &opts,
        temp_ptr,
        temp_bytes,
        out_ptr,
        out_bytes,
        stream);
    throwExceptionIfError(status, "nvcompBitcompCompressAsync failed");
  }

  int m_algorithm;
};

/**
 * @brief Bitcomp decompressor offered by nvcomp
 *
 * @tparam T The type to decompress to.
 */
template <typename T>
class BitcompDecompressor : public Decompressor<T>
{

public:
  BitcompDecompressor(
      const void* const compressed_data,
      const size_t compressed_data_size,
      cudaStream_t stream) :
      Decompressor<T>(compressed_data, compressed_data_size, stream),
      expected_uncompressed_size(0)
  {
    // Sync the stream to make sure the compressed data is available
    cudaStreamSynchronize(stream);

    // Create a plan from the compressed data itself
    if (bitcompCreatePlanFromCompressedData(&handle, compressed_data)
        != BITCOMP_SUCCESS)
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Bitcomp decompressor: Plan creation failed");

    if (bitcompGetUncompressedSizeFromHandle(
            handle, &expected_uncompressed_size)
        != BITCOMP_SUCCESS)
      throw NVCompException(
          nvcompErrorInternal,
          "Bitcomp decompressor: can't get uncompressed size from handle");

    // Make sure the type matches the compressed data
    bitcompDataType_t expectedDatatype = BITCOMP_UNSIGNED_8BIT;
    if (std::is_same<T, int8_t>::value)
      expectedDatatype = BITCOMP_SIGNED_8BIT;
    if (std::is_same<T, uint16_t>::value)
      expectedDatatype = BITCOMP_UNSIGNED_16BIT;
    if (std::is_same<T, int16_t>::value)
      expectedDatatype = BITCOMP_SIGNED_16BIT;
    if (std::is_same<T, uint32_t>::value)
      expectedDatatype = BITCOMP_UNSIGNED_32BIT;
    if (std::is_same<T, int32_t>::value)
      expectedDatatype = BITCOMP_SIGNED_32BIT;
    if (std::is_same<T, uint64_t>::value)
      expectedDatatype = BITCOMP_UNSIGNED_64BIT;
    if (std::is_same<T, int64_t>::value)
      expectedDatatype = BITCOMP_SIGNED_64BIT;
    bitcompDataType_t dataType;
    if (bitcompGetDataTypeFromHandle(handle, &dataType) != BITCOMP_SUCCESS)
      throw NVCompException(
          nvcompErrorInternal,
          "Bitcomp decompressor: can't get type from handle");
    if (expectedDatatype != dataType)
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Bitcomp decompressor: data type doesn't match");

    // Make sure the compressed size matches
    size_t expected_size;
    if (bitcompGetCompressedSize(compressed_data, &expected_size)
        != BITCOMP_SUCCESS)
      throw NVCompException(
          nvcompErrorInternal,
          "Bitcomp decompressor: can't get compressed size");
    if (expected_size != compressed_data_size)
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Bitcomp decompressor: compressed size doesn't match");

    // until we can access it from parent class
    m_compressed_data = compressed_data;
  }

  ~BitcompDecompressor()
  {
    if (bitcompDestroyPlan(handle) != BITCOMP_SUCCESS)
      throw NVCompException(
          nvcompErrorInvalidValue, "Bitcomp decompressor: Invalid plan");
  }

  // disable copying
  BitcompDecompressor(const BitcompDecompressor& other) = delete;
  BitcompDecompressor& operator=(const BitcompDecompressor& other) = delete;

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
      void* const temp_ptr,
      const size_t temp_bytes,
      T* const out_ptr,
      const size_t out_num_elements,
      cudaStream_t stream)
  {
    if (expected_uncompressed_size > out_num_elements * sizeof(T))
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Bitcomp decompressor: Output buffer too small");
    if (bitcompSetStream(handle, stream) != BITCOMP_SUCCESS)
      throw NVCompException(
          nvcompErrorInternal, "Bitcomp: Invalid handle or stream");
    // if (bitcompUncompress(handle, this->get_compressed_data(), out_ptr)
    if (bitcompUncompress(handle, m_compressed_data, out_ptr)
        != BITCOMP_SUCCESS)
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Bitcomp decompressor: Invalid handle or stream");
  }

  /**
   * @brief Get the size of the temporary buffer required for decompression.
   *
   * @return The size in bytes.
   */
  size_t get_temp_size()
  {
    return 0;
  }

  /**
   * @brief Get the size of the output buffer in bytes.
   *
   * @return The size in bytes.
   */
  size_t get_output_size()
  {
    return expected_uncompressed_size;
  }

  /**
   * @brief Get the number of elements that will be decompressed.
   *
   * @return The number of elements.
   */
  size_t get_num_elements()
  {
    return (expected_uncompressed_size / sizeof(T));
  }

private:
  bitcompHandle_t handle;
  size_t expected_uncompressed_size;
  const void* m_compressed_data; // Can't access the one from Decompressor
};

} // namespace nvcomp

#endif // ENABLE_BITCOMP

#endif