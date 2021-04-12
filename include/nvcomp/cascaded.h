/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVCOMP_CASCADED_H
#define NVCOMP_CASCADED_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure that stores the compression configuration
 */
typedef struct
{
  /**
   * @brief The number of Run Length Encodings to perform.
   */
  int num_RLEs;

  /**
   * @brief The number of Delta Encodings to perform.
   */
  int num_deltas;

  /**
   * @brief Whether or not to bitpack the final layers.
   */
  int use_bp;
} nvcompCascadedFormatOpts;

/**
 * @brief Configure the Cascaded compressor and return temp and output
 * sizes needed to perform the compression.
 *
 * @param in_ptr The compressed memory on the device.
 * @param in_bytes The size of the compressed memory on the device.
 * @param in_type The data type of the uncompressed data.
 * @param format_opts The cascaded format options.
 * @param temp_bytes The temporary memory required for compression (output)
 * @param out_bytes The estaimted size of the compressed result (output)
 * @param metadata_bytes The bytes needed to store the metadata (output)
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompCascadedCompressConfigure(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    const nvcompCascadedFormatOpts* format_opts,
    void** metadata_ptr,
    size_t* temp_bytes,
    size_t* out_bytes,
    size_t* metadata_bytes);

/**
 * @brief Perform asynchronous compression. The pointer `out_bytes` must be to
 * pinned memory for this to be asynchronous.
 *
 * NOTE: Currently, cascaded compression is limited to 2^31-1 bytes. To
 * compress larger data, break it up into chunks.
 *
 * @param in_ptr The uncompressed data on the device.
 * @param in_bytes The size of the uncompressed data in bytes.
 * @param in_type The data type of the uncompressed data.
 * @param format_opts The cascaded format options.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param out_ptr The location to write compresesd data to on the device.
 * @param out_bytes The size of the output location on input, and the size of
 * the compressed data on output. If pinned memory, the stream must be
 * synchronized with, before reading.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompCascadedCompressAsync(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    const nvcompCascadedFormatOpts* format_opts,
    void* temp_ptr,
    size_t temp_bytes,
    void* out_ptr,
    size_t* out_bytes,
    cudaStream_t stream);

/**
 * @brief Extract the metadata from the compressed data.
 *
 * @param out_ptr The compressed data.
 * @param out_bytes The size of the compressed data in bytes.
 * @param metadata_ptr Pointer to the resulting metadata (output)
 * @param metadata_bytes The size of the resulting metadata.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompLZ4QueryMetadataAsync(
    const void * out_ptr,
    size_t out_bytes,
    void * metadata_ptr,
    size_t metadata_bytes,
    cudaStream_t stream);


/**
 * @brief Destroys the metadata object and frees the associated memory.
 *
 * @param metadata_ptr The pointer to destroy.
 */
void nvcompCascadedDestroyMetadata(void* metadata_ptr);


/**
 * @brief Configure the decompression and get the output and temp sizes
 * needed to perform the decompression.  
 *
 * NOTE: Is synchronoous if metadata is not explicitly given, as it needs
 * to be extracted from the compressed data.
 * NOTE: Currently, cascaded compression is limited to 2^31-1 bytes. To
 * compress larger data, break it up into chunks.
 *
 * @param in_ptr The compressed data on the device.
 * @param in_bytes The size of the compressed data in bytes.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param out_bytes The required size of the output location in bytes (output).
 * @param metadata_ptr Pointer to metadata accessible on the host.  If null,
 * metadata is automatically extracted from the compressed bitstream, requiring
 * device synchronization, and this pointer is set to the extracted metadata.
 * @param metadata_bytes The size of the metadata.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompCascadedDecompressConifugre(
    const void* in_ptr,
    size_t in_bytes,
    size_t* temp_bytes,
    size_t* out_bytes,
    void* metadata_ptr,
    size_t* metadata_bytes,
    cudaStream_t stream);


/**
 * @brief Perform the asynchronous decompression.
 *
 * @param in_ptr The compressed data on the device to decompress.
 * @param in_bytes The size of the compressed data.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace.
 * @param metadata_ptr The metadata (accessible by host).
 * @param out_ptr The output location on the device.
 * @param out_bytes The size of the output location.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompCascadedDecompressAsync(
    const void* in_ptr,
    size_t in_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    const void* metadata_ptr,
    void* out_ptr,
    size_t out_bytes,
    cudaStream_t stream);

/**************************************************************************
 *  Automatically configured API calls
 *************************************************************************/
/**
 * @brief Configure the Cascaded compressor using auto-selector and return temp and output
 * sizes needed to perform the compression.
 *
 * @param in_ptr The compressed memory on the device.
 * @param in_bytes The size of the compressed memory on the device.
 * @param in_type The data type of the uncompressed data.
 * @param temp_bytes The temporary memory required for compression (output)
 * @param out_bytes The estaimted size of the compressed result (output)
 * @param metadata_bytes The bytes needed to store the metadata (output)
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompCascadedCompressAutoConfigure(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    void** metadata_ptr,
    size_t* temp_bytes,
    size_t* out_bytes,
    size_t* metadata_bytes);

/**
 * @brief Perform compression by first determining the best configuration for
 * the input.  Runs synchronously because we first find the configuration to
 * use. Random sampling is performed and selected using random numbers generated
 * using the seed provided.
 *
 * NOTE: Currently, cascaded compression is limited to 2^31-1 bytes. To
 * compress larger data, break it up into chunks.
 *
 * @param in_ptr The uncompressed data on the device.
 * @param in_bytes The size of the uncompressed data in bytes.
 * @param in_type The data type of the uncompressed data.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param out_ptr The location to write compresesd data to on the device.
 * @param out_bytes The size of the output location on input, and the size of
 * the compressed data on output. If pinned memory, the stream must be
 * synchronized with, before reading.
 * @param seed The seed used for the random generator used in sampling
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompCascadedCompressAuto(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    void* temp_ptr,
    size_t temp_bytes,
    void* out_ptr,
    size_t* out_bytes,
    unsigned seed,
    cudaStream_t stream);

/**************************************************************************
 *  Cascaded Selector types and API calls
 *************************************************************************/

/**
 * @brief Structure that stores options to run Cascaded Selector
 * NOTE: Minimum values for both parameters is 1, maximum for
 * sample_size is 1024 and is allso limited by the input size:
 *        (sample_size * num_samples) <= input_size
 */
typedef struct
{
  /**
   * @brief The number of elements used in each sample
   * minimum value 1, maximum value 1024
   */
  size_t sample_size;

  /**
   * @brief The number of samples used by the selector
   * minimum value 1
   */
  size_t num_samples;

  /**
   * @brief The seed used for the random sampling
   */
  unsigned seed;

} nvcompCascadedSelectorOpts;

/**
 * @brief Configure the cascaded selector and get the temp and output sizes need
 * to run the cascaded selector.
 *
 * @param in_bytes The size of the uncompressed data in bytes.
 * @param in_type The data type of the uncompressed data.
 * @param opts The configuration options for the selector.
 * @param temp_bytes The size of the temporary workspace in bytes (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompCascadedSelectorConfigure(
    size_t in_bytes,
    nvcompType_t in_type,
    nvcompCascadedSelectorOpts opts,
    size_t* temp_bytes);

/**
 * @brief Run the cascaded selector to determine the best cascaded compression
 * configuration and estimated compression ratio.
 *
 * @param in_ptr The uncompressed data on the device.
 * @param in_bytes The size of the uncompressed data in bytes.
 * @param in_type The data type of the uncompressed data.
 * @param opts The configuration options for the selector.
 * @param temp_ptr The temporary workspace memory on the device
 * @param temp_bytes The size of the temporary workspace in bytes
 * @param format_opts The best cascaded compression configuration (output)
 * @param est_ratio The estimated compression ratio using the configuration
 * (output)
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompCascadedSelectorRun(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    nvcompCascadedSelectorOpts opts,
    void* temp_ptr,
    size_t temp_bytes,
    nvcompCascadedFormatOpts* format_opts,
    double* est_ratio,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
