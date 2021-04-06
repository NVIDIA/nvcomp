/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


namespace nvcomp {

/**
 * @brief The result of the compression and decompression routines
 **/
struct gpu_snappy_status_s {
  uint32_t status; // Non-zero value indicates an error
};

/**
 * @brief Interface for compressing data with Snappy
 *
 * The function compresses multiple independent chunks of data.
 * All the pointers parameters are to GPU-accessible memory.
 *
 * @param[in] device_in_ptr Pointer to the list of pointers to
 * the GPU-accessible uncompressed data.
 * @param[in] device_in_bytes Pointer to the list of sizes of uncompressed
 * data
 * @param[in] device_out_ptr Pointer to the buffer with pointers,
 * where the function should put compressed data to.
 * @param[in] device_out_available_bytes Pointer to the list of sizes of
 * memory chunks referenced by device_out_ptr. Could be null-ptr indicating
 * all output buffers has enough size to store compressed data.
 * @param[out] outputs Pointer to the statuses of compression for each chunk.
 * Could be null-ptr.
 * @param[out] device_out_bytes Pointer to the list of actual sizes
 * of compressed data.
 * @param[in] count The number of chunks to compress.
 * @param[in] stream All the compression will be enqueued into this CUDA
 * stream and run asynchronously.
 **/
cudaError_t gpu_snap(
  const void* const* device_in_ptr,
	const size_t* device_in_bytes,
	void* const* device_out_ptr,
	const size_t* device_out_available_bytes,
	gpu_snappy_status_s *outputs,
	size_t* device_out_bytes,
  int count,
  cudaStream_t stream);

/**
 * @brief Interface for decompressing data with Snappy
 *
 * The function decompresses multiple independent chunks of data.
 * All the pointers parameters are to GPU-accessible memory.
 *
 * @param[in] device_in_ptr Pointer to the kist of pointers to
 * the GPU-accessible compressed data.
 * @param[in] device_in_bytes Pointer to the list of sizes of compressed
 * data.
 * @param[in] device_out_ptr Pointer to the buffer with pointers,
 * where the function should put uncompressed data to.
 * @param[in] device_out_available_bytes Pointer to the list of sizes of
 * memory chunks referenced by device_out_ptr. Could be null-ptr indicating
 * all output buffers has enough size to stored uncompressed data.
 * @param[out] outputs Pointer to the statuses of decompression for each chunk.
 * Could be null-ptr.
 * @param[out] device_out_bytes Pointer to the list of actual sizes
 * of uncompressed data. Could be null-ptr.
 * @param[in] count The number of chunks to decompress.
 * @param[in] stream All the decompression will be enqueued into this CUDA
 * stream and run asynchronously.
 **/
cudaError_t gpu_unsnap(
  const void* const* device_in_ptr,
  const size_t* device_in_bytes,
  void* const* device_out_ptr,
  const size_t* device_out_available_bytes,
  gpu_snappy_status_s *outputs,
  size_t* device_out_bytes,
  int count,
  cudaStream_t stream);
} // namespace nvcomp
