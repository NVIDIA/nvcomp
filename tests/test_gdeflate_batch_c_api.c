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

#include "nvcomp/gdeflate.h"

#include "cuda_runtime.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Test GPU compression and decompression using the gdeflate chunked API //

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

#define BATCH_SIZE 1130

  // prepare input and output on host
  size_t batch_sizes_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_sizes_host[i] = ((i * 1103) % ((1<<16)/sizeof(T)-2500)) + 2500;
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

  nvcompStatus_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedGdeflateCompressGetTempSize(
      BATCH_SIZE, max_chunk_size, &comp_temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_comp_out_bytes;
  status = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
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

  status = nvcompBatchedGdeflateCompressAsync(
      (const void* const*)d_in_ptrs,
      batch_bytes_device,
      max_chunk_size,
      BATCH_SIZE,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes_device,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    cudaFree(host_in_ptrs[i]);
  }
  cudaFree(d_in_ptrs);

  // Gdeflate decompression does not need temp space.
  size_t temp_bytes;
  status = nvcompBatchedGdeflateDecompressGetTempSize(
      BATCH_SIZE, max_chunk_size, &temp_bytes);

  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

  void* host_decomp_out[BATCH_SIZE];
  for (int i = 0; i < BATCH_SIZE; i++) {
    CUDA_CHECK(cudaMalloc(&host_decomp_out[i], batch_bytes_host[i]));
  }
  void** d_decomp_out;
  cudaMalloc((void**)&d_decomp_out, sizeof(*d_decomp_out) * BATCH_SIZE);
  cudaMemcpy(
      d_decomp_out,
      host_decomp_out,
      sizeof(*d_decomp_out) * BATCH_SIZE,
      cudaMemcpyHostToDevice);

  status = nvcompBatchedGdeflateDecompressAsync(
      (const void* const*)d_comp_out,
      comp_out_bytes_device,
      batch_bytes_device,
      max_chunk_size,
      BATCH_SIZE,
      temp_ptr,
      temp_bytes,
      (void* const*)d_decomp_out,
      stream);

  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(batch_bytes_device);
  cudaFree(comp_out_bytes_device);
  cudaFree(temp_ptr);

  for (int i = 0; i < BATCH_SIZE; i++) {
    cudaMemcpy(
        output_host[i],
        host_decomp_out[i],
        batch_bytes_host[i],
        cudaMemcpyDeviceToHost);
    // Verify correctness
    for (size_t j = 0; j < batch_bytes_host[i] / sizeof(T); ++j) {
      REQUIRE(output_host[i][j] == input_host[i][j]);
    }
  }

  for (int i = 0; i < BATCH_SIZE; i++) {
    cudaFree(host_comp_out[i]);
    cudaFree(host_decomp_out[i]);
    free(output_host[i]);
  }
  cudaFree(d_comp_out);

  return 1;

#undef BATCH_SIZE
}

int main(int argc, char** argv)
{
  if (argc != 1) {
    printf("ERROR: %s accepts no arguments.\n", argv[0]);
    return 1;
  }

  int rv = 0;

#ifdef ENABLE_GDEFLATE
  int num_tests = 1;

  if (!test_batch_compression_and_decompression()) {
    printf("compression and decompression test failed.\n");
    rv += 1;
  }

  if (rv == 0) {
    printf("SUCCESS: All tests passed: %d/%d\n", (num_tests - rv), num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", rv, num_tests);
  }
#endif // ENABLE_GDEFLATE

  return rv;
}
