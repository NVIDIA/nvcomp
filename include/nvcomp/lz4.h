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

#ifndef NVCOMP_LZ4_H
#define NVCOMP_LZ4_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure for configuring LZ4 compression.
 */
typedef struct
{
  /**
   * @brief The size of each chunk of data to decompress indepentently with
   * LZ4. Must be within the range of [32768, 16777216]. Larger sizes will
   * result in higher compression, but with decreased parallelism. The
   * recommended size is 65536.
   */
  size_t chunk_size;
} nvcompLZ4FormatOpts;

/**
 * @brief Check if a given chunk of compressed data on the GPU is LZ4.
 *
 * @param in_ptr The compressed data.
 * @param in_bytes The size of the compressed data.
 * @param stream The stream to fetch data from the GPU on.
 *
 * @return 1 If the data is compressed via LZ4.
 */
int LZ4IsData(const void* const in_ptr, size_t in_bytes, cudaStream_t stream);

/**
 * @brief Check if the given CPU-accessible metadata is for LZ4.
 *
 * @param metadata_ptr The metadata pointer.
 *
 * @return 1 if the data is for LZ4.
 */
int LZ4IsMetadata(const void* const metadata_ptr);

/**
 * @brief Configure an LZ4 compressor and return temp and output sizes needed
 * to perform the compression. If no format is provided (i.e., NULL),
 * the default options will be used.
 *
 * @param format_opts The LZ4 format options, may be NULL to use defaults.
 * @param type The data type of the uncompressed data.
 * @param uncompressed_bytes The size of the uncompressed data on the device.
 * @param metadata_bytes The bytes needed to store the metadata (output)
 * @param temp_bytes The temporary memory required for compression (output)
 * @param compressed_bytes The estaimted size of the compressed result (output)
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompLZ4CompressConfigure(
    const nvcompLZ4FormatOpts* format_opts,
    nvcompType_t in_type,
    size_t uncompresed_bytes,
    size_t* metadata_bytes,
    size_t* temp_bytes,
    size_t* max_compressed_bytes);

/**
 * @brief Perform asynchronous compression. The pointers `compressed_ptr` and
 * `compressed_bytes` must be to preallocated memory directly accessible by the
 * GPU. If no format is provided (i.e., NULL), the default options will be used.
 *
 * @param format_opts The LZ4 options to use, must match those passed to
 * nvcompLZ4CompressConfigure(). This can be null to use the default options.
 * @param in_type The type being compressed.
 * @param uncompressed_ptr The uncompressed data on the device.
 * @param uncompressed_bytes The size of the compressed data in bytes.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param compressed_ptr The output location on the device.
 * @param compressed_bytes The size of the compressed data (output). This must
 * be GPU accessible.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompLZ4CompressAsync(
    const nvcompLZ4FormatOpts* format_opts,
    const nvcompType_t in_type,
    const void* const uncompressed_ptr,
    const size_t uncompressed_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const compressed_ptr,
    size_t* compressed_bytes,
    cudaStream_t stream);

/**
 * @brief Configure the decompression and get the output and temp sizes
 * needed to perform the decompression. This function allocates host-side
 * memory, synchronizes the provided CUDA stream, and blocks CPU execution until
 * the metadata is extracted and copied from the `compressed_ptr`.
 *
 * @param compressed_ptr The compressed data on the device.
 * @param compressed_bytes The size of the compressed data in bytes.
 * @param metadata_ptr The pointer that is to be populated with the metadata
 * needed to perform decompression.  This function allocates host-side memory
 * and copies the metdata to it.
 * @param metadata_bytes The size of the metadata that this function allocates.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param uncompressed_bytes The required size of the output location in bytes
 * (output).
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompLZ4DecompressConfigure(
    const void* compressed_ptr,
    size_t compressed_bytes,
    void** metadata_ptr,
    size_t* metadata_bytes,
    size_t* temp_bytes,
    size_t* uncompressed_bytes,
    cudaStream_t stream);

/**
 * @brief Perform the asynchronous decompression.
 *
 * @param compressed_ptr The compressed data on the device.
 * @param compressed_bytes The size of the compressed data.
 * @param metadata_ptr The metadata (accessible by host).
 * @param metadata_bytes The size of the metadata.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace.
 * @param uncompressed_ptr The location to decompress data to on the GPU
 * (output).
 * @param uncompressed_bytes The size of the uncompressed data as returned by
 * `nvcompLZ4DecompressConfigure()`.
 * @param stream THe stream to decompress on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompLZ4DecompressAsync(
    const void* compressed_ptr,
    size_t compressed_bytes,
    const void* metadata_ptr,
    size_t metadata_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    void* uncompressed_ptr,
    size_t uncompressed_bytes,
    cudaStream_t stream);

/**
 * @brief Destroys the metadata object and frees the associated memory.  Must be
 * used to destroy metadata that is generated from nvcompLZ4DecompressConfigure.
 *
 * @param metadata_ptr The pointer to destroy.
 */
void nvcompLZ4DestroyMetadata(void* metadata_ptr);

/******************************************************************************
 * Batched compression/decompression interface
 *****************************************************************************/

/**
 * @brief Get temporary space required for compression.
 *
 * Chunk size must not exceed
 * 16777216 bytes. For best performance, a chunk size of 65536 bytes is
 * recommended.
 *
 * @param batch_size The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the
 * batch.
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedLZ4CompressGetTempSize(
    size_t batch_size, size_t max_uncompressed_chunk_bytes, size_t* temp_bytes);

/**
 * @brief Get the maximum size any chunk could compress to in the batch. That
 * is, the minimum amount of output memory required to be given
 * nvcompBatchedLZ4CompressAsync() for each batch item.
 *
 * Chunk size must not exceed
 * 16777216 bytes. For best performance, a chunk size of 65536 bytes is
 * recommended.
 *
 * @param max_uncompressed_chunk_bytes The maximum size of a chunk in the batch.
 * @param max_compressed_byes The maximum compressed size of the largest chunk
 * (output).
 *
 * @return The nvcompSuccess unless there is an error.
 */
nvcompError_t nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
    size_t max_uncompressed_chunk_bytes, size_t* max_compressed_bytes);

/**
 * @brief Perform compression asynchronously. All pointers must point to GPU
 * accessible locations. The individual chunk size must not exceed
 * 16777216 bytes. For best performance, a chunk size of 65536 bytes is
 * recommended.
 *
 * @param device_in_ptr The pointers on the GPU, to uncompressed batched items.
 * This pointer must be GPU accessible.
 * @param device_in_bytes The size of each uncompressed batch item on the GPU.
 * @param max_uncompressed_chunk_bytes The maximum size in bytes of the largest
 * chunk in the batch. This parameter is currently unused, so if it is not set
 * with the maximum size, it should be set to zero. If a future version makes
 * use of it, it will return an error if it is set to zero.
 * @param batch_size The number of batch items.
 * @param device_temp_ptr The temporary GPU workspace.
 * @param temp_bytes The size of the temporary GPU workspace.
 * @param device_out_ptr The pointers on the GPU, to the output location for
 * each compressed batch item (output). This pointer must be GPU accessible.
 * @param device_out_bytes The compressed size of each chunk on the GPU
 * (output). This pointer must be GPU accessible.
 * @param stream The stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
nvcompError_t nvcompBatchedLZ4CompressAsync(
    const void* const* device_in_ptr,
    const size_t* device_in_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_out_ptr,
    size_t* device_out_bytes,
    cudaStream_t stream);

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The size of the largest chunk in bytes
 * when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedLZ4DecompressGetTempSize(
    size_t num_chunks, size_t max_uncompressed_chunk_bytes, size_t* temp_bytes);

/**
 * @brief Perform decompression asynchronously. All pointers must be GPU
 * accessible.
 *
 * @param device_in_ptrs The pointers on the GPU, to the compressed chunks.
 * This pointer must be accessible from the GPU.
 * @param device_in_bytes The size of each compressed chunk on the GPU.
 * @param device_out_bytes The size of each uncompressed chunk on the GPU.
 * @param max_uncompressed_chunk_bytes The maximum size in bytes of the largest
 * chunk in the batch. This parameter is currently unused, so if it is not set
 * with the maximum size, it should be set to zero. If a future version makes
 * use of it, it will return an error if it is set to zero.
 * @param batch_size The number of batch items.
 * @param device_temp_ptr The temporary GPU space.
 * @param temp_bytes The size of the temporary GPU space.
 * @param device_out_ptr The pointers on the GPU, to where to uncompress each
 * chunk (output). This pointer must be accessible from the GPU.
 * @param stream The stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedLZ4DecompressAsync(
    const void* const* device_in_ptrs,
    const size_t* device_in_bytes,
    const size_t* device_out_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    const size_t temp_bytes,
    void* const* device_out_ptr,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
