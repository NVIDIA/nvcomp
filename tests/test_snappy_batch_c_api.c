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

#if defined(_WIN32)
#include <malloc.h>
#endif

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

int test_batch_compression_and_decompression(const size_t batch_size)
{
  typedef int T;

  // set a constant seed
  srand(0);

  // prepare input and output on host
  size_t* batch_sizes_host = malloc(batch_size * sizeof(size_t));

  for (size_t i = 0; i < batch_size; ++i) {
    batch_sizes_host[i] = ((i * 1103) % 100000) + 10000;
  }

  size_t* batch_bytes_host = malloc(batch_size * sizeof(size_t));

  size_t max_batch_bytes_uncompressed = 0; 
  for (size_t i = 0; i < batch_size; ++i) {
    batch_bytes_host[i] = sizeof(T) * batch_sizes_host[i];
    if (batch_bytes_host[i] > max_batch_bytes_uncompressed)
      max_batch_bytes_uncompressed = batch_bytes_host[i];
  }

  size_t * batch_bytes_device;
  CUDA_CHECK(cudaMalloc((void **)(&batch_bytes_device), batch_size * sizeof(*batch_bytes_host)));
  cudaMemcpy(batch_bytes_device, batch_bytes_host, batch_size * sizeof(*batch_bytes_host), cudaMemcpyHostToDevice);

  T** input_host = malloc(batch_size * sizeof(T*));

  for (size_t i = 0; i < batch_size; ++i) {
    input_host[i] = malloc(sizeof(T) * batch_sizes_host[i]);
    for (size_t j = 0; j < batch_sizes_host[i]; ++j) {
      // make sure there should be some repeats to compress
      input_host[i][j] = (rand() % 4) + 300;
    }
  }

  T** output_host = malloc(batch_size * sizeof(T*));

  for (size_t i = 0; i < batch_size; ++i) {
    output_host[i] = malloc(batch_bytes_host[i]);
  }

  // prepare gpu buffers
  void** d_in_data = malloc(batch_size * sizeof(void*));

  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&d_in_data[i], batch_bytes_host[i]));
    CUDA_CHECK(cudaMemcpy(
        d_in_data[i],
        input_host[i],
        batch_bytes_host[i],
        cudaMemcpyHostToDevice));
  }
  void** d_in_data_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_in_data_device), batch_size * sizeof(*d_in_data)));
  cudaMemcpy(d_in_data_device, d_in_data, batch_size * sizeof(*d_in_data), cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);


  nvcompError_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedSnappyCompressGetTempSize(
      batch_size,
      max_batch_bytes_uncompressed,
      &comp_temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t comp_out_bytes;
  status = nvcompBatchedSnappyCompressGetOutputSize(
      max_batch_bytes_uncompressed,
      &comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void** d_comp_out = malloc(batch_size * sizeof(void*));

  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&d_comp_out[i], comp_out_bytes));
  }

  void** d_comp_out_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_comp_out_device), batch_size * sizeof(*d_comp_out)));
  cudaMemcpy(d_comp_out_device, d_comp_out, batch_size * sizeof(*d_comp_out), cudaMemcpyHostToDevice);

  size_t * comp_out_bytes_device;
  CUDA_CHECK(cudaMalloc((void **)(&comp_out_bytes_device), sizeof(size_t *) * batch_size));

  status = nvcompBatchedSnappyCompressAsync(
      (const void* const*)d_in_data_device,
      batch_bytes_device,
      batch_size,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out_device,
      comp_out_bytes_device,
      stream);
  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  cudaFree(d_in_data_device);
  for (size_t i = 0; i < batch_size; ++i) {
    cudaFree(d_in_data[i]);
  }

  // Snappy decompression does not need temp space.
  size_t temp_bytes;
  status = nvcompBatchedSnappyDecompressGetTempSize(
      batch_size,
      max_batch_bytes_uncompressed,
      &temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

  void** d_decomp_out = malloc(batch_size * sizeof(void*));
  for (size_t i = 0; i < batch_size; i++) {
    CUDA_CHECK(cudaMalloc(&d_decomp_out[i], max_batch_bytes_uncompressed));
  }
  void** d_decomp_out_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_decomp_out_device), batch_size * sizeof(*d_decomp_out)));
  cudaMemcpy(d_decomp_out_device, d_decomp_out, batch_size * sizeof(*d_decomp_out), cudaMemcpyHostToDevice);

  status = nvcompBatchedSnappyDecompressAsync(
      (const void* const*)d_comp_out_device,
      comp_out_bytes_device,
      batch_bytes_device,
      batch_size,
      temp_ptr,
      temp_bytes,
      (void* const*)d_decomp_out_device,
      stream);
  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(temp_ptr);
  cudaFree(d_comp_out_device);
  cudaFree(comp_out_bytes_device);
  cudaFree(batch_bytes_device);

  for (size_t i = 0; i < batch_size; i++) {
    cudaMemcpy(
        output_host[i],
        d_decomp_out[i],
        batch_bytes_host[i],
        cudaMemcpyDeviceToHost);
    // Verify correctness
    for (size_t j = 0; j < batch_bytes_host[i] / sizeof(T); ++j) {
      REQUIRE(output_host[i][j] == input_host[i][j]);
    }
  }

  for (size_t i = 0; i < batch_size; i++) {
    cudaFree(d_comp_out[i]);
    cudaFree(d_decomp_out[i]);
    free(output_host[i]);
    free(input_host[i]);
  }

  free(batch_sizes_host);
  free(batch_bytes_host);
  free(input_host);
  free(output_host);
  free(d_in_data);
  free(d_comp_out);
  free(d_decomp_out);

  return 1;
}

int main(int argc, char** argv)
{
  if (argc != 1) {
    printf("ERROR: %s accepts no arguments.\n", argv[0]);
    return 1;
  }

  size_t rv = 0;

  size_t batch_sizes[] = {1130, 920, 2700};
  size_t num_tests = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

  for (size_t i = 0; i < sizeof(batch_sizes) / sizeof(batch_sizes[0]); ++i) {
    if (!test_batch_compression_and_decompression(batch_sizes[i])) {
      printf("compression and decompression test failed.\n");
      rv += 1;
    }
  }

  if (rv == 0) {
    printf("SUCCESS: All tests passed: %zd/%zd\n", (num_tests - rv), num_tests);
  } else {
    printf("FAILURE: %zd/%zd tests failed\n", rv, num_tests);
  }

  return rv;
}
