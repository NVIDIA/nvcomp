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

#include "lz4.h"

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

int test_compression(void)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  nvcompLZ4FormatOpts comp_opts = {1 << 16};

  // set a constant seed
  srand(0);

  const size_t batch_size = 11;

  // prepare input and output on host
  const size_t batchSizesHost[11]
      = {163, 123, 90000, 1029, 103218, 564, 178, 92, 1011, 9124, 1024};

  size_t batchBytesHost[11];
  for (size_t i = 0; i < batch_size; ++i) {
    batchBytesHost[i] = sizeof(T) * batchSizesHost[i];
  }

  T* inputHost[11];
  for (size_t i = 0; i < batch_size; ++i) {
    inputHost[i] = malloc(sizeof(T) * batchSizesHost[i]);
    for (size_t j = 0; j < batchSizesHost[i]; ++j) {
      // make sure there should be some repeats to compress
      inputHost[i][j] = (rand() % 10) + 300;
    }
  }

  size_t outputBytesHost[11];
  T* outputHost[11];
  for (size_t i = 0; i < batch_size; ++i) {
    outputHost[i] = malloc(batchBytesHost[i]);
  }

  // prepare gpu buffers
  void* d_in_data[11];
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&d_in_data[i], batchBytesHost[i]));
    CUDA_CHECK(cudaMemcpy(
        d_in_data[i], inputHost[i], batchBytesHost[i], cudaMemcpyHostToDevice));
  }
  void* d_out_data[11];
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&d_out_data[i], batchBytesHost[i]));
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompError_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedLZ4CompressGetTempSize(
      (const void* const*)d_in_data,
      batchBytesHost,
      batch_size,
      &comp_opts,
      &comp_temp_bytes);
  REQUIRE(status == cudaSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t comp_out_bytes[11];
  status = nvcompBatchedLZ4CompressGetOutputSize(
      (const void* const*)d_in_data,
      batchBytesHost,
      batch_size,
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      comp_out_bytes);
  REQUIRE(status == cudaSuccess);

  void* d_comp_out[11];
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&d_comp_out[i], comp_out_bytes[i]));
  }

  status = nvcompBatchedLZ4CompressAsync(
      (const void* const*)d_in_data,
      batchBytesHost,
      batch_size,
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes,
      stream);
  REQUIRE(status == cudaSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  for (size_t i = 0; i < batch_size; ++i) {
    cudaFree(d_in_data[i]);
  }

  // perform decompression with separate calls
  for (size_t i = 0; i < batch_size; ++i) {
    // select compression algorithm
    // Get metadata
    void* metadata_ptr;
    status = nvcompDecompressGetMetadata(
        d_comp_out[i], comp_out_bytes[i], &metadata_ptr, stream);
    REQUIRE(status == cudaSuccess);

    // get temp size
    size_t temp_bytes;
    status = nvcompDecompressGetTempSize(metadata_ptr, &temp_bytes);
    REQUIRE(status == cudaSuccess);

    // allocate temp buffer
    void* temp_ptr;
    CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

    // get output size
    size_t output_bytes;
    status = nvcompDecompressGetOutputSize(metadata_ptr, &output_bytes);
    REQUIRE(status == cudaSuccess);

    // allocate output buffer
    void* out_ptr;
    CUDA_CHECK(cudaMalloc(&out_ptr, output_bytes));

    // execute decompression (asynchronous)
    status = nvcompDecompressAsync(
        d_comp_out[i],
        comp_out_bytes[i],
        temp_ptr,
        temp_bytes,
        metadata_ptr,
        out_ptr,
        output_bytes,
        stream);
    REQUIRE(status == cudaSuccess);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    nvcompDecompressDestroyMetadata(metadata_ptr);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(
        outputHost[i], out_ptr, output_bytes, cudaMemcpyDeviceToHost));

    cudaFree(temp_ptr);
    cudaFree(d_comp_out[i]);

    // Verify correctness
    for (size_t j = 0; j < batchSizesHost[i]; ++j) {
      REQUIRE(outputHost[i][j] == inputHost[i][j]);
    }
    free(outputHost[i]);
    free(inputHost[i]);
  }

  return 1;
}

int main(int argc, char** argv)
{
  int num_tests = 2;
  int rv = 0;

  if (!test_compression()) {
    printf("compression test failed.");
    rv += 1;
  }

  if (rv == 0) {
    printf("SUCCESS: All tests passed: %d/%d\n", (num_tests - rv), num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", rv, num_tests);
  }

  return rv;
}
