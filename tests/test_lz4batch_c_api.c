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

#include "nvcomp/lz4.h"

#include "cuda_runtime.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Test GPU decompression with cascaded compression API //

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      return 0;                                                                \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      printf(                                                                  \
          "API call failure \"" #func "\" with %d at " __FILE__ ":%d\n",       \
          (int)rt,                                                             \
          __LINE__);                                                           \
      return 0;                                                                \
    }                                                                          \
  } while (0)

int test_batch_compression_and_decompression(void)
{
  typedef int T;

  // set a constant seed
  srand(0);

  const size_t BATCH_SIZE = 1130;

  // prepare input and output on host
  size_t batch_sizes_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_sizes_host[i] = ((i * 1103) % 100000) + 10000;
  }

  size_t batch_bytes_host[BATCH_SIZE];
  size_t max_chunk_size = 0;
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_bytes_host[i] = sizeof(T) * batch_sizes_host[i];
    if (batch_bytes_host[i] > max_chunk_size) {
      max_chunk_size = batch_bytes_host[i];
    }
  }

  T* input_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    input_host[i] = malloc(sizeof(T) * batch_sizes_host[i]);
    for (size_t j = 0; j < batch_sizes_host[i]; ++j) {
      // make sure there should be some repeats to compress
      input_host[i][j] = (rand() % 4) + 300;
    }
  }

  T* output_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    output_host[i] = malloc(batch_bytes_host[i]);
  }

  // prepare gpu buffers
  void* host_in_ptrs[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&host_in_ptrs[i], batch_bytes_host[i]));
    CUDA_CHECK(cudaMemcpy(
        host_in_ptrs[i],
        input_host[i],
        batch_bytes_host[i],
        cudaMemcpyHostToDevice));
  }
  void** d_in_ptrs;
  CUDA_CHECK(cudaMalloc((void**)&d_in_ptrs, sizeof(*d_in_ptrs) * BATCH_SIZE));
  CUDA_CHECK(cudaMemcpy(
      d_in_ptrs,
      host_in_ptrs,
      sizeof(*d_in_ptrs) * BATCH_SIZE,
      cudaMemcpyHostToDevice));

  size_t* batch_bytes_device;
  CUDA_CHECK(cudaMalloc(
      (void**)&batch_bytes_device, sizeof(*batch_bytes_device) * BATCH_SIZE));
  CUDA_CHECK(cudaMemcpy(
      batch_bytes_device,
      batch_bytes_host,
      sizeof(*batch_bytes_device) * BATCH_SIZE,
      cudaMemcpyHostToDevice));

  nvcompError_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedLZ4CompressGetTempSize(
      BATCH_SIZE, max_chunk_size, &comp_temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_comp_out_bytes;
  status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      max_chunk_size, &max_comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void* host_comp_out[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&host_comp_out[i], max_comp_out_bytes));
  }
  void** d_comp_out;
  CUDA_CHECK(cudaMalloc((void**)&d_comp_out, sizeof(*d_comp_out) * BATCH_SIZE));
  cudaMemcpy(
      d_comp_out,
      host_comp_out,
      sizeof(*d_comp_out) * BATCH_SIZE,
      cudaMemcpyHostToDevice);

  size_t* comp_out_bytes_device;
  cudaMalloc(
      (void**)&comp_out_bytes_device,
      sizeof(*comp_out_bytes_device) * BATCH_SIZE);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcomp_lz4_lowlevel_opt_type fmt_opts = {0};

  status = nvcompBatchedLZ4CompressAsync(
      (const void* const*)d_in_ptrs,
      batch_bytes_device,
      max_chunk_size,
      BATCH_SIZE,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes_device,
      &fmt_opts,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    cudaFree(host_in_ptrs[i]);
  }
  cudaFree(d_in_ptrs);

  size_t temp_bytes;
  status = nvcompBatchedLZ4DecompressGetTempSize(
      BATCH_SIZE, max_chunk_size, &temp_bytes);

  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

  void* host_decomp_out[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; i++) {
    CUDA_CHECK(cudaMalloc(&host_decomp_out[i], batch_bytes_host[i]));
  }
  void** d_decomp_out;
  cudaMalloc((void**)&d_decomp_out, sizeof(*d_decomp_out) * BATCH_SIZE);

  size_t* device_actual_uncompressed_bytes;
  cudaMallocManaged(
      (void**)&device_actual_uncompressed_bytes,
      sizeof(*device_actual_uncompressed_bytes) * BATCH_SIZE,
      cudaMemAttachGlobal);

  nvcompStatus_t* device_status_ptrs;
  cudaMallocManaged(
      (void**)&device_status_ptrs,
      sizeof(*device_status_ptrs) * BATCH_SIZE,
      cudaMemAttachGlobal);

  cudaMemcpy(
      d_decomp_out,
      host_decomp_out,
      sizeof(*d_decomp_out) * BATCH_SIZE,
      cudaMemcpyHostToDevice);

  status = nvcompBatchedLZ4DecompressAsync(
      (const void* const*)d_comp_out,
      comp_out_bytes_device,
      batch_bytes_device,
      device_actual_uncompressed_bytes,
      BATCH_SIZE,
      temp_ptr,
      temp_bytes,
      (void* const*)d_decomp_out,
      device_status_ptrs,
      stream);

  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(batch_bytes_device);
  cudaFree(comp_out_bytes_device);
  cudaFree(temp_ptr);

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    cudaMemcpy(
        output_host[i],
        host_decomp_out[i],
        batch_bytes_host[i],
        cudaMemcpyDeviceToHost);
    // Verify correctness
    for (size_t j = 0; j < batch_bytes_host[i] / sizeof(T); ++j) {
      REQUIRE(output_host[i][j] == input_host[i][j]);
    }
    REQUIRE(device_actual_uncompressed_bytes[i] == batch_bytes_host[i]);
    REQUIRE(device_status_ptrs[i] == nvcompStatusSuccess);
  }

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    cudaFree(host_comp_out[i]);
    cudaFree(host_decomp_out[i]);
    free(output_host[i]);
  }
  cudaFree(d_comp_out);
  cudaFree(device_actual_uncompressed_bytes);
  cudaFree(device_status_ptrs);

  return 1;
}

int test_batch_compression_and_decompression_zero_max(void)
{
  // This test is meant to be exactly the same as the above, except is provides
  // `0` as the maximum chunk size to the compress and decompress methods,
  // where it supposed to be unused. We promise in our docs, that if we make
  // use of it, we will check that it is not zero.

  typedef int T;

  // set a constant seed
  srand(0);

  const size_t BATCH_SIZE = 1130;

  // prepare input and output on host
  size_t batch_sizes_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_sizes_host[i] = ((i * 1103) % 100000) + 10000;
  }

  size_t batch_bytes_host[BATCH_SIZE];
  size_t max_chunk_size = 0;
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_bytes_host[i] = sizeof(T) * batch_sizes_host[i];
    if (batch_bytes_host[i] > max_chunk_size) {
      max_chunk_size = batch_bytes_host[i];
    }
  }

  T* input_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    input_host[i] = malloc(sizeof(T) * batch_sizes_host[i]);
    for (size_t j = 0; j < batch_sizes_host[i]; ++j) {
      // make sure there should be some repeats to compress
      input_host[i][j] = (rand() % 4) + 300;
    }
  }

  T* output_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    output_host[i] = malloc(batch_bytes_host[i]);
  }

  // prepare gpu buffers
  void* host_in_ptrs[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&host_in_ptrs[i], batch_bytes_host[i]));
    CUDA_CHECK(cudaMemcpy(
        host_in_ptrs[i],
        input_host[i],
        batch_bytes_host[i],
        cudaMemcpyHostToDevice));
  }
  void** d_in_ptrs;
  CUDA_CHECK(cudaMalloc((void**)&d_in_ptrs, sizeof(*d_in_ptrs) * BATCH_SIZE));
  CUDA_CHECK(cudaMemcpy(
      d_in_ptrs,
      host_in_ptrs,
      sizeof(*d_in_ptrs) * BATCH_SIZE,
      cudaMemcpyHostToDevice));

  size_t* batch_bytes_device;
  CUDA_CHECK(cudaMalloc(
      (void**)&batch_bytes_device, sizeof(*batch_bytes_device) * BATCH_SIZE));
  CUDA_CHECK(cudaMemcpy(
      batch_bytes_device,
      batch_bytes_host,
      sizeof(*batch_bytes_device) * BATCH_SIZE,
      cudaMemcpyHostToDevice));

  nvcompError_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedLZ4CompressGetTempSize(
      BATCH_SIZE, max_chunk_size, &comp_temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_comp_out_bytes;
  status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      max_chunk_size, &max_comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void* host_comp_out[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&host_comp_out[i], max_comp_out_bytes));
  }
  void** d_comp_out;
  CUDA_CHECK(cudaMalloc((void**)&d_comp_out, sizeof(*d_comp_out) * BATCH_SIZE));
  cudaMemcpy(
      d_comp_out,
      host_comp_out,
      sizeof(*d_comp_out) * BATCH_SIZE,
      cudaMemcpyHostToDevice);

  size_t* comp_out_bytes_device;
  cudaMalloc(
      (void**)&comp_out_bytes_device,
      sizeof(*comp_out_bytes_device) * BATCH_SIZE);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcomp_lz4_lowlevel_opt_type fmt_opts = {0};

  status = nvcompBatchedLZ4CompressAsync(
      (const void* const*)d_in_ptrs,
      batch_bytes_device,
      0,
      BATCH_SIZE,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes_device,
      &fmt_opts,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    cudaFree(host_in_ptrs[i]);
  }
  cudaFree(d_in_ptrs);

  size_t temp_bytes;
  status = nvcompBatchedLZ4DecompressGetTempSize(
      BATCH_SIZE, max_chunk_size, &temp_bytes);

  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

  void* host_decomp_out[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; i++) {
    CUDA_CHECK(cudaMalloc(&host_decomp_out[i], batch_bytes_host[i]));
  }
  void** d_decomp_out;
  cudaMalloc((void**)&d_decomp_out, sizeof(*d_decomp_out) * BATCH_SIZE);
  cudaMemcpy(
      d_decomp_out,
      host_decomp_out,
      sizeof(*d_decomp_out) * BATCH_SIZE,
      cudaMemcpyHostToDevice);

  size_t* device_actual_uncompressed_bytes;
  cudaMallocManaged(
      (void**)&device_actual_uncompressed_bytes,
      sizeof(*device_actual_uncompressed_bytes) * BATCH_SIZE,
      cudaMemAttachGlobal);

  nvcompStatus_t* device_status_ptrs;
  cudaMallocManaged(
      (void**)&device_status_ptrs,
      sizeof(*device_status_ptrs) * BATCH_SIZE,
      cudaMemAttachGlobal);

  status = nvcompBatchedLZ4DecompressAsync(
      (const void* const*)d_comp_out,
      comp_out_bytes_device,
      batch_bytes_device,
      device_actual_uncompressed_bytes,
      BATCH_SIZE,
      temp_ptr,
      temp_bytes,
      (void* const*)d_decomp_out,
      device_status_ptrs,
      stream);

  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(batch_bytes_device);
  cudaFree(comp_out_bytes_device);
  cudaFree(temp_ptr);

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    cudaMemcpy(
        output_host[i],
        host_decomp_out[i],
        batch_bytes_host[i],
        cudaMemcpyDeviceToHost);
    // Verify correctness
    for (size_t j = 0; j < batch_bytes_host[i] / sizeof(T); ++j) {
      REQUIRE(output_host[i][j] == input_host[i][j]);
    }
    REQUIRE(device_actual_uncompressed_bytes[i] == batch_bytes_host[i]);
    REQUIRE(device_status_ptrs[i] == nvcompStatusSuccess);
  }

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    cudaFree(host_comp_out[i]);
    cudaFree(host_decomp_out[i]);
    free(output_host[i]);
  }
  cudaFree(d_comp_out);
  cudaFree(device_actual_uncompressed_bytes);
  cudaFree(device_status_ptrs);

  return 1;
}

/**
 * Tests calculating the decompression sizes when we don't know them.
 * Performs a dry-run of the LZ4 decompression kernel to extract them.
 */
int test_batch_get_decomp_sizes()
{

  typedef int T;

  const size_t BATCH_SIZE = 1130;

  // prepare input and output on host
  size_t batch_sizes_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_sizes_host[i] = ((i * 1103) % 100000) + 10000;
  }

  size_t batch_bytes_host[BATCH_SIZE];
  size_t max_chunk_size = 0;
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_bytes_host[i] = sizeof(T) * batch_sizes_host[i];
    if (batch_bytes_host[i] > max_chunk_size) {
      max_chunk_size = batch_bytes_host[i];
    }
  }

  T** in_ptrs;
  CUDA_CHECK(cudaMallocManaged(
      (void**)&in_ptrs, sizeof(*in_ptrs) * BATCH_SIZE, cudaMemAttachGlobal));
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMallocManaged(
        (void**)&in_ptrs[i], batch_bytes_host[i], cudaMemAttachGlobal));
    for (size_t j = 0; j < batch_sizes_host[i]; ++j) {
      // make sure there should be some repeats to compress
      in_ptrs[i][j] = (rand() % 4) + 300;
    }
  }

  size_t* batch_bytes_device;
  CUDA_CHECK(cudaMalloc(
      (void**)&batch_bytes_device, sizeof(*batch_bytes_device) * BATCH_SIZE));
  CUDA_CHECK(cudaMemcpy(
      batch_bytes_device,
      batch_bytes_host,
      sizeof(*batch_bytes_device) * BATCH_SIZE,
      cudaMemcpyHostToDevice));

  nvcompError_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedLZ4CompressGetTempSize(
      BATCH_SIZE, max_chunk_size, &comp_temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_comp_out_bytes;
  status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      max_chunk_size, &max_comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void* host_comp_out[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&host_comp_out[i], max_comp_out_bytes));
  }
  void** d_comp_out;
  CUDA_CHECK(cudaMalloc((void**)&d_comp_out, sizeof(*d_comp_out) * BATCH_SIZE));
  cudaMemcpy(
      d_comp_out,
      host_comp_out,
      sizeof(*d_comp_out) * BATCH_SIZE,
      cudaMemcpyHostToDevice);

  size_t* comp_out_bytes_device;
  cudaMallocManaged(
      (void**)&comp_out_bytes_device,
      sizeof(*comp_out_bytes_device) * BATCH_SIZE,
      cudaMemAttachGlobal);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcomp_lz4_lowlevel_opt_type fmt_opts = {0};

  status = nvcompBatchedLZ4CompressAsync(
      (const void* const*)in_ptrs,
      batch_bytes_device,
      max_chunk_size,
      BATCH_SIZE,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes_device,
      &fmt_opts,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    cudaFree(in_ptrs[i]);
  }
  cudaFree(in_ptrs);
  cudaFree(d_comp_temp);
  cudaFree(batch_bytes_device);

  size_t* decompressed_sizes;
  CUDA_CHECK(cudaMallocManaged(
      (void**)&decompressed_sizes,
      sizeof(size_t) * BATCH_SIZE,
      cudaMemAttachGlobal));

  status = nvcompBatchedLZ4GetDecompressSizeAsync(
      (const void* const*)d_comp_out,
      comp_out_bytes_device,
      decompressed_sizes,
      BATCH_SIZE,
      stream);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    // Verify calculated decompressed sizes are same as the original
    REQUIRE(decompressed_sizes[i] == batch_bytes_host[i]);
  }

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    cudaFree(host_comp_out[i]);
  }
  cudaFree(d_comp_out);
  cudaFree(decompressed_sizes);
  cudaFree(comp_out_bytes_device);

  return 1;
}

/**
 * Negative test of decompressing invalid data to ensure we fail gracefully and
 * output the expected error status and sizes
 */
int test_cannot_decompress()
{
  typedef int T;

  const size_t BATCH_SIZE = 1130;

  // prepare input and output on host
  size_t batch_bytes_host[BATCH_SIZE];
  size_t max_chunk_size = 0;
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_bytes_host[i] = (((i * 1103) % 100000) + 10000) * sizeof(T);
    if (batch_bytes_host[i] > max_chunk_size) {
      max_chunk_size = batch_bytes_host[i];
    }
  }

  size_t* compressed_sizes;
  CUDA_CHECK(cudaMallocManaged(
      (void**)&compressed_sizes,
      sizeof(size_t) * BATCH_SIZE,
      cudaMemAttachGlobal));
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    // Let's just assume 2x compression
    compressed_sizes[i] = batch_bytes_host[i] / 2;
    assert(compressed_sizes[i] > 0);
  }

  size_t* decompressed_sizes;
  CUDA_CHECK(cudaMallocManaged(
      (void**)&decompressed_sizes,
      sizeof(size_t) * BATCH_SIZE,
      cudaMemAttachGlobal));

  T** compressed_ptrs;
  CUDA_CHECK(cudaMallocManaged(
      (void**)&compressed_ptrs,
      sizeof(*compressed_ptrs) * BATCH_SIZE,
      cudaMemAttachGlobal));
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMallocManaged(
        (void**)&compressed_ptrs[i], batch_bytes_host[i], cudaMemAttachGlobal));
    for (size_t j = 0; j < batch_bytes_host[i] / sizeof(T); ++j) {
      compressed_ptrs[i][j] = rand(); // corrupted LZ4 sequences
    }
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Test getting the sizes of a corrupt stream to ensure this doesn't crash
  // Unfortunately we cannot detect a LZ4 corrupt stream so the size will just
  // be garbage instead of 0
  nvcompError_t status = nvcompBatchedLZ4GetDecompressSizeAsync(
      (const void* const*)compressed_ptrs,
      compressed_sizes,
      decompressed_sizes,
      BATCH_SIZE,
      stream);

  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream)); // Shouldn't crash

  // Allocate buffers now that we know the decompressed sizes
  void** decompressed_ptrs;
  CUDA_CHECK(cudaMallocManaged(
      (void**)&decompressed_ptrs,
      sizeof(*decompressed_ptrs) * BATCH_SIZE,
      cudaMemAttachGlobal));
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMallocManaged(
        (void**)&decompressed_ptrs[i],
        decompressed_sizes[i],
        cudaMemAttachGlobal));
  }

  size_t temp_bytes;
  status = nvcompBatchedLZ4DecompressGetTempSize(
      BATCH_SIZE, max_chunk_size, &temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_temp;
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

  nvcompStatus_t* device_status_ptrs;
  cudaMallocManaged(
      (void**)&device_status_ptrs,
      sizeof(*device_status_ptrs) * BATCH_SIZE,
      cudaMemAttachGlobal);

  // Test decompressing corrupted sequences
  status = nvcompBatchedLZ4DecompressAsync(
      (const void* const*)compressed_ptrs,
      compressed_sizes,
      decompressed_sizes, // Decompress using calculated sizes
      decompressed_sizes, // It's okay to overwrite, there is no race here
      BATCH_SIZE,
      d_temp,
      temp_bytes,
      (void* const*)decompressed_ptrs,
      device_status_ptrs,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream)); // Shouldn't crash

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    // Size should be 0 indicating it failed to decompress
    REQUIRE(decompressed_sizes[i] == 0);
    REQUIRE(device_status_ptrs[i] == nvcompStatusCannotDecompress);
  }

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    cudaFree(compressed_ptrs[i]);
    cudaFree(decompressed_ptrs[i]);
  }
  cudaFree(compressed_ptrs);
  cudaFree(compressed_sizes);
  cudaFree(decompressed_sizes);
  cudaFree(decompressed_ptrs);
  cudaFree(d_temp);
  cudaFree(device_status_ptrs);

  return 1;
}

int main(int argc, char** argv)
{
  if (argc != 1) {
    printf("ERROR: %s accepts no arguments.\n", argv[0]);
    return 1;
  }

  // Set constant seed
  srand(0);

  int num_tests = 4;
  int rv = 0;

  if (!test_batch_compression_and_decompression()) {
    printf("compression and decompression test failed.\n");
    rv += 1;
  }

  if (!test_batch_compression_and_decompression_zero_max()) {
    printf("compression and decompression zero_max test failed.\n");
    rv += 1;
  }

  if (!test_batch_get_decomp_sizes()) {
    printf("get decompression sizes test failed.\n");
    rv += 1;
  }

  if (!test_cannot_decompress()) {
    printf("get negative decompression sizes test failed.\n");
    rv += 1;
  }

  if (rv == 0) {
    printf("SUCCESS: All tests passed: %d/%d\n", (num_tests - rv), num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", rv, num_tests);
  }

  return rv;
}
