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

#include "nvcomp/snappy.h"

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

#define BATCH_SIZE 1130

  // prepare input and output on host
  size_t batch_sizes_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_sizes_host[i] = ((i * 1103) % 100000) + 10000;
  }

  size_t batch_bytes_host[BATCH_SIZE];
  size_t max_batch_bytes_host = 0; 
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_bytes_host[i] = sizeof(T) * batch_sizes_host[i];
    if (batch_bytes_host[i] > max_batch_bytes_host)
      max_batch_bytes_host = batch_bytes_host[i];
  }

  size_t * batch_bytes_device;
  CUDA_CHECK(cudaMalloc((void **)(&batch_bytes_device), sizeof(batch_bytes_host)));
  cudaMemcpy(batch_bytes_device, batch_bytes_host, sizeof(batch_bytes_host), cudaMemcpyHostToDevice);

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
  void* d_in_data[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&d_in_data[i], batch_bytes_host[i]));
    CUDA_CHECK(cudaMemcpy(
        d_in_data[i],
        input_host[i],
        batch_bytes_host[i],
        cudaMemcpyHostToDevice));
  }
  void** d_in_data_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_in_data_device), sizeof(d_in_data)));
  cudaMemcpy(d_in_data_device, d_in_data, sizeof(d_in_data), cudaMemcpyHostToDevice);


  void* d_out_data[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&d_out_data[i], batch_bytes_host[i]));
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompError_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedSnappyCompressGetTempSize(
      BATCH_SIZE,
      max_batch_bytes_host,
      &comp_temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t comp_out_bytes;
  status = nvcompBatchedSnappyCompressGetOutputSize(
      max_batch_bytes_host,
      &comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_out[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&d_comp_out[i], comp_out_bytes));
  }

  void** d_comp_out_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_comp_out_device), sizeof(d_comp_out)));
  cudaMemcpy(d_comp_out_device, d_comp_out, sizeof(d_comp_out), cudaMemcpyHostToDevice);

  size_t * comp_out_bytes_device;
  CUDA_CHECK(cudaMalloc((void **)(&comp_out_bytes_device), sizeof(size_t *) * BATCH_SIZE));

  status = nvcompBatchedSnappyCompressAsync(
      (const void* const*)d_in_data_device,
      batch_bytes_device,
      BATCH_SIZE,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out_device,
      comp_out_bytes_device,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    cudaFree(d_in_data[i]);
  }

  void* metadata_ptr;

  status = nvcompBatchedSnappyDecompressGetMetadata(
      (const void**)d_comp_out,
      comp_out_bytes,
      BATCH_SIZE,
      &metadata_ptr,
      stream);

  // Snappy decompression does not need temp space.
  size_t temp_bytes;
  status = nvcompBatchedSnappyDecompressGetTempSize(
      (const void*)metadata_ptr, &temp_bytes);

  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

  size_t decomp_out_bytes[BATCH_SIZE];
  status = nvcompBatchedSnappyDecompressGetOutputSize(
      (const void*)metadata_ptr, BATCH_SIZE, decomp_out_bytes);

  void* d_decomp_out[BATCH_SIZE];
  for (int i = 0; i < BATCH_SIZE; i++) {
    CUDA_CHECK(cudaMalloc(&d_decomp_out[i], decomp_out_bytes[i]));
  }

  status = nvcompBatchedSnappyDecompressAsync(
      (const void* const*)d_comp_out,
      comp_out_bytes,
      BATCH_SIZE,
      temp_ptr,
      temp_bytes,
      metadata_ptr,
      (void* const*)d_decomp_out,
      decomp_out_bytes,
      stream);

  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaDeviceSynchronize());

  nvcompBatchedSnappyDecompressDestroyMetadata(metadata_ptr);
  cudaFree(temp_ptr);

  for (int i = 0; i < BATCH_SIZE; i++) {
    cudaMemcpy(
        output_host[i],
        d_decomp_out[i],
        decomp_out_bytes[i],
        cudaMemcpyDeviceToHost);
    // Verify correctness
    for (size_t j = 0; j < decomp_out_bytes[i] / sizeof(T); ++j) {
      REQUIRE(output_host[i][j] == input_host[i][j]);
    }
  }

  for (int i = 0; i < BATCH_SIZE; i++) {
    cudaFree(d_comp_out[i]);
    cudaFree(d_decomp_out[i]);
    free(output_host[i]);
  }

  return 1;

#undef BATCH_SIZE
}

int main(int argc, char** argv)
{
  if (argc != 1) {
    printf("ERROR: %s accepts no arguments.\n", argv[0]);
    return 1;
  }

  int num_tests = 1;
  int rv = 0;

  if (!test_batch_compression_and_decompression()) {
    printf("compression and decompression test failed.\n");
    rv += 1;
  }

  if (rv == 0) {
    printf("SUCCESS: All tests passed: %d/%d\n", (num_tests - rv), num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", rv, num_tests);
  }

  return rv;
}
