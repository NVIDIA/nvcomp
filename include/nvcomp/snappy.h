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

#ifndef NVCOMP_SNAPPY_H
#define NVCOMP_SNAPPY_H

#include "nvcomp.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**************************************************************************
 *  - Experimental - Subject to change -
 * Batched compression/decompression interface for Snappy
 * ************************************************************************/

/**
 * @brief Extracts the metadata from all the input baches in_ptr on the device
 * and copies them to the host. This function synchronizes on the stream.
 *
 * @param in_ptr Array of compressed chunks on the device.
 * @param in_bytes Array of sizes of the compressed chunks on the device.
 * @param batch_size Number of chunks in the batch (cardinality of in_bytes and
 * in_ptr)
 * @param metadata_ptr The batch of metadata on the host to create from all the
 * compresesed data chunks in the batch.
 * @param stream The stream to use for reading memory from the device.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedSnappyDecompressGetMetadata(
    const void** in_ptr,
    const size_t* in_bytes,
    size_t batch_size,
    void** metadata_ptr,
    cudaStream_t stream);

/**
 * @brief Destroys metadata and frees the associated memory.
 *
 * @para metadata_ptr List of metadata to destroy.
 */
void nvcompBatchedSnappyDecompressDestroyMetadata(void* metadata_ptr);

/**
 * @brief Computes the temporary storage size needed to decompress the batch of
 * data.
 *
 * @param metadata_ptr The metadata for all compressed chunks in the batch.
 * @param temp_bytes The size of temporary workspace required to perform
 * decomrpession of all chunks in the batch, in bytes (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedSnappyDecompressGetTempSize(
    const void* metadata_ptr, size_t* temp_bytes);

/**
 * @brief Computes the decompressed size of each chunk of in the batch.
 *
 * @param metadata_ptr The metadata for all compressed chunks.
 * @param batch_size The number of chunks in the batch (cardinality of
 * output_bytes).
 * @param output_bytes Array of sizes of the decompressed data in bytes
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedSnappyDecompressGetOutputSize(
    const void* metadata_ptr, size_t batch_size, size_t* output_bytes);

/**
 * @brief Perform the asynchronous decompression on batch of compressed chunks
 * of data.
 *
 * @param in_ptr Array of compressed data chunks on the device to decompress.
 * @param in_bytes The sizes of each chunk of compressed data.
 * @param batch_size The number of chunks in the batch (cardinality of other
 * inputs).
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace.
 * @param metadata_ptr The metadata of all chunks in the batch.
 * @param out_ptr The output location on the device.
 * @param out_bytes The sizes of each decompressed chunk.
 * @param stream The cuda stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedSnappyDecompressAsync(
    const void* const* in_ptr,
    const size_t* in_bytes,
    size_t batch_size,
    void* const temp_ptr,
    const size_t temp_bytes,
    const void* metadata_ptr,
    void* const* out_ptr,
    const size_t* out_bytes,
    cudaStream_t stream);

/**
 * @brief Get temporary space required for compression.
 *
 * @param batch_size The number of items in the batch.
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompError_t nvcompBatchedSnappyCompressGetTempSize(
    size_t batch_size,
    size_t max_chunk_size,
    size_t * temp_bytes);

/**
 * @brief Get the maximum size any chunk could compress to in the batch. That is, the minimum amount of output memory required to be given nvcompBatchedLZ4CompressAsync() for each batch item.
 *
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param max_compressed_size The maximum compressed size of the largest chunk (output).
 *
 * @return The nvcompSuccess unless there is an error.
 */
nvcompError_t nvcompBatchedSnappyCompressGetOutputSize(
    size_t max_chunk_size,
    size_t * max_compressed_size);

/**
 * @brief Perform compression.
 *
 * @param device_in_ptr The pointers on the GPU, to uncompressed batched items.
 * @param device_in_bytes The size of each uncompressed batch item on the GPU.
 * @param batch_size The number of batch items.
 * @param temp_ptr The temporary GPU workspace.
 * @param temp_bytes The size of the temporary GPU workspace.
 * @param device_out_ptr The pointers on the GPU, to the output location for each compressed batch item (output).
 * @param device_out_bytes The compressed size of each chunk on the GPU (output).
 * @param stream The stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
nvcompError_t nvcompBatchedSnappyCompressAsync(
	const void* const* device_in_ptr,
	const size_t* device_in_bytes,
	size_t batch_size,
	void* temp_ptr,
	size_t temp_bytes,
	void* const* device_out_ptr,
	size_t* device_out_bytes,
	cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
