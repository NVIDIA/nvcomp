/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "lowlevel/SnappyBatchKernels.h"
#include "SnappyKernels.cuh"
#include "CudaUtils.h"

namespace nvcomp {

/**
 * @brief Snappy compression kernel
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * blockDim {64,1,1}
 *
 * @param[in] inputs Source/Destination buffer information per block
 * @param[out] outputs Compression status per block
 * @param[in] count Number of blocks to compress
 **/
__global__ void __launch_bounds__(64)
snap_kernel(
  const void* const* __restrict__ device_in_ptr,
  const uint64_t* __restrict__ device_in_bytes,
  void* const* __restrict__ device_out_ptr,
  const uint64_t* __restrict__ device_out_available_bytes,
  gpu_snappy_status_s * __restrict__ outputs,
	uint64_t* device_out_bytes)
{
  const int ix_chunk = blockIdx.x;
  do_snap(reinterpret_cast<const uint8_t*>(device_in_ptr[ix_chunk]),
      device_in_bytes[ix_chunk],
      reinterpret_cast<uint8_t*>(device_out_ptr[ix_chunk]),
      device_out_available_bytes ? device_out_available_bytes[ix_chunk] : 0,
      outputs ? &outputs[ix_chunk] : nullptr,
      &device_out_bytes[ix_chunk]);
}

__global__ void __launch_bounds__(32)
get_uncompressed_sizes_kernel(
  const void* const* __restrict__ device_in_ptr,
  const uint64_t* __restrict__ device_in_bytes,
	uint64_t* __restrict__ device_out_bytes)
{
  int t             = threadIdx.x;
  int strm_id       = blockIdx.x;

  if (t == 0) {
    uint32_t uncompressed_size = 0;
    const uint8_t *cur = reinterpret_cast<const uint8_t *>(device_in_ptr[strm_id]);
    const uint8_t *end = cur + device_in_bytes[strm_id];
    if (cur < end) {
      // Read uncompressed size (varint), limited to 31-bit
      // The size is stored as little-endian varint, from 1 to 5 bytes (as we allow up to 2^31 sizes only)
      // The upper bit of each byte indicates if there is another byte to read to compute the size
      // Please see format details at https://github.com/google/snappy/blob/master/format_description.txt 
      uncompressed_size = *cur++;
      if (uncompressed_size > 0x7f) {
        uint32_t c        = (cur < end) ? *cur++ : 0;
        uncompressed_size = (uncompressed_size & 0x7f) | (c << 7);
        // Check if the most significant bit is set, this indicates we need to read the next byte
        // (maybe even more) to compute the uncompressed size
        // We do it several time stopping if 1) MSB is cleared or 2) we see that the size is >= 2^31
        // which we cannot handle  
        if (uncompressed_size >= (0x80 << 7)) {
          c                 = (cur < end) ? *cur++ : 0;
          uncompressed_size = (uncompressed_size & ((0x7f << 7) | 0x7f)) | (c << 14);
          if (uncompressed_size >= (0x80 << 14)) {
            c = (cur < end) ? *cur++ : 0;
            uncompressed_size =
              (uncompressed_size & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 21);
            if (uncompressed_size >= (0x80 << 21)) {
              c = (cur < end) ? *cur++ : 0;
              // Snappy format alllows uncompressed sizes larger than 2^31
              // We generate an error in this case
              if (c < 0x8)
                uncompressed_size =
                  (uncompressed_size & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) |
                  (c << 28);
              else
                uncompressed_size = 0;
            }
          }
        }
      }
    }
    device_out_bytes[strm_id] = uncompressed_size;
  }
}

/**
 * @brief Snappy decompression kernel
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * blockDim {96,1,1}
 *
 * @param[in] inputs Source & destination information per block
 * @param[out] outputs Decompression status per block
 **/
__global__ void __launch_bounds__(96) unsnap_kernel(
    const void* const* __restrict__ device_in_ptr,
    const uint64_t* __restrict__ device_in_bytes,
    void* const* __restrict__ device_out_ptr,
    const uint64_t* __restrict__ device_out_available_bytes,
    nvcompStatus_t* const __restrict__ outputs,
    uint64_t* __restrict__ device_out_bytes)
{
  const int ix_chunk = blockIdx.x;
  do_unsnap(reinterpret_cast<const uint8_t*>(device_in_ptr[ix_chunk]),
      device_in_bytes[ix_chunk],
      reinterpret_cast<uint8_t*>(device_out_ptr[ix_chunk]),
      device_out_available_bytes ? device_out_available_bytes[ix_chunk] : 0,
      outputs ? &outputs[ix_chunk] : nullptr,
      device_out_bytes ? &device_out_bytes[ix_chunk] : nullptr);
}

void gpu_snap(
  const void* const* device_in_ptr,
	const size_t* device_in_bytes,
	void* const* device_out_ptr,
	const size_t* device_out_available_bytes,
	gpu_snappy_status_s *outputs,
	size_t* device_out_bytes,
  int count,
  cudaStream_t stream)
{
  dim3 dim_block(64, 1);  // 2 warps per stream, 1 stream per block
  dim3 dim_grid(count, 1);
  if (count > 0) { snap_kernel<<<dim_grid, dim_block, 0, stream>>>(
    device_in_ptr, device_in_bytes, device_out_ptr, device_out_available_bytes,
      outputs, device_out_bytes); }
  CudaUtils::check_last_error("Failed to launch Snappy compression CUDA kernel gpu_snap");
}

void gpu_unsnap(
    const void* const* device_in_ptr,
    const size_t* device_in_bytes,
    void* const* device_out_ptr,
    const size_t* device_out_available_bytes,
    nvcompStatus_t* outputs,
    size_t* device_out_bytes,
    int count,
    cudaStream_t stream)
{
  uint32_t count32 = (count > 0) ? count : 0;
  dim3 dim_block(96, 1);     // 3 warps per stream, 1 stream per block
  dim3 dim_grid(count32, 1);  // TODO: Check max grid dimensions vs max expected count

  unsnap_kernel<<<dim_grid, dim_block, 0, stream>>>(
    device_in_ptr, device_in_bytes, device_out_ptr, device_out_available_bytes,
      outputs, device_out_bytes);
  CudaUtils::check_last_error("Failed to launch Snappy decompression CUDA kernel gpu_unsnap");
}

void gpu_get_uncompressed_sizes(
  const void* const* device_in_ptr,
  const size_t* device_in_bytes,
  size_t* device_out_bytes,
  int count,
  cudaStream_t stream)
{
  dim3 dim_block(32, 1);
  dim3 dim_grid(count, 1);

  get_uncompressed_sizes_kernel<<<dim_grid, dim_block, 0, stream>>>(
    device_in_ptr, device_in_bytes, device_out_bytes);
  CudaUtils::check_last_error("Failed to run Snappy kernel gpu_get_uncompressed_sizes");
}

} // nvcomp namespace
