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

#ifndef NVCOMP_BITCOMP_H
#define NVCOMP_BITCOMP_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef ENABLE_BITCOMP

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure for configuring Bitcomp compression.
 */
typedef struct
{
  /**
   * @brief Bitcomp algorithm options.
   *  algorithm_type: The type of Bitcomp algorithm used.
   *    0 : Default algorithm, usually gives the best compression ratios
   *    1 : "Sparse" algorithm, works well on sparse data (with lots of zeroes).
   *        and is usually a faster than the default algorithm.
   */
  int algorithm_type;
} nvcompBitcompFormatOpts;

/**
 * @brief Get the temporary workspace size required to perform compression.
 *
 * @param format_opts The bitcomp format options (can pass NULL for default
 * options).
 * @param in_type The type of the uncompressed data.
 * @param uncompressed_bytes The size of the uncompressed data in bytes.
 * @param temp_bytes The size of the required temporary workspace in bytes
 * (output).
 * @param max_compressed_bytes The maximum size of the compressed data
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompCompressConfigure(
    const nvcompBitcompFormatOpts* opts,
    nvcompType_t in_type,
    size_t in_bytes,
    size_t* metadata_bytes,
    size_t* temp_bytes,
    size_t* max_compressed_bytes);

/**
 * @brief Perform asynchronous compression.
 *
 * @param format_opts The bitcomp format options (can pass NULL for default
 * options).
 * @param in_type The data type of the uncompressed data.
 * @param uncompressed_ptr The uncompressed data on the device.
 * @param uncompressed_bytes The size of the uncompressed data in bytes.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param compressed_ptr The location to write compresesd data to on the device
 * (output).
 * @param compressed_bytes The size of the compressed data (output). This must
 * be GPU accessible.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompCompressAsync(
    const nvcompBitcompFormatOpts* format_opts,
    nvcompType_t in_type,
    const void* uncompressed_ptr,
    size_t uncompressed_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    void* compressed_ptr,
    size_t* compressed_bytes,
    cudaStream_t stream);

/**
 * @brief Extracts the metadata from the input in_ptr on the device and copies
 * it to the host. This function synchronizes on the stream.
 *
 * @param compressed_ptr The compressed memory on the device.
 * @param compressed_bytes The size of the compressed memory on the device.
 * @param metadata_ptr The metadata on the host to create from the compresesd
 * data (output).
 * @param metadata_bytes The size of the created metadata (output).
 * @param temp_bytes The amount of temporary space required for decompression
 * (output).
 * @param uncompressed_bytes The size the data will decompress to (output).
 * @param stream The stream to use for copying from the device to the host.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompDecompressConfigure(
    const void* compressed_ptr,
    size_t compressed_bytes,
    void** metadata_ptr,
    size_t* metadata_bytes,
    size_t* temp_bytes,
    size_t* uncompressed_bytes,
    cudaStream_t stream);

/**
 * @brief Destroys the metadata object and frees the associated memory.
 *
 * @param metadata_ptr The pointer to destroy.
 */
void nvcompBitcompDestroyMetadata(void* metadata_ptr);

/**
 * @brief Perform the asynchronous decompression.
 *
 * @param compressed_ptr The compressed data on the device to decompress.
 * @param compressed_bytes The size of the compressed data.
 * @param metadata_ptr The metadata.
 * @param metadata_bytes The size of the metadata.
 * @param temp_ptr The temporary workspace on the device. Not used, can pass
 * NULL.
 * @param temp_bytes The size of the temporary workspace. Not used.
 * @param uncompressed_ptr The output location on the device.
 * @param uncompressed_bytes The size of the output location.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBitcompDecompressAsync(
    const void* compressed_ptr,
    size_t compressed_bytes,
    void* metadata_ptr,
    size_t metadata_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    void* uncompressed_ptr,
    size_t uncompressed_bytes,
    cudaStream_t stream);

/**
 * @brief Checks if the compressed data was compressed with bitcomp.
 *
 * @param in_ptr The compressed data.
 * @param in_bytes The size of the compressed buffer.
 *
 * @return 1 if the data was compressed with bitcomp, 0 otherwise
 */
int nvcompIsBitcompData(const void* const in_ptr, size_t in_bytes);

#ifdef __cplusplus
}
#endif

#endif  // ENABLE_BITCOMP

#endif
