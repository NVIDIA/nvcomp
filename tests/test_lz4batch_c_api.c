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

int test_batch_compression(void)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  nvcompLZ4FormatOpts comp_opts = {1 << 16};

  // set a constant seed
  srand(0);

#define BATCH_SIZE 11

  // prepare input and output on host
  const size_t batch_sizes_host[BATCH_SIZE]
      = {163, 123, 90000, 1029, 103218, 564, 178, 92, 1011, 9124, 1024};

  size_t batch_bytes_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_bytes_host[i] = sizeof(T) * batch_sizes_host[i];
  }

  T* input_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    input_host[i] = malloc(sizeof(T) * batch_sizes_host[i]);
    for (size_t j = 0; j < batch_sizes_host[i]; ++j) {
      // make sure there should be some repeats to compress
      input_host[i][j] = (rand() % 10) + 300;
    }
  }

  size_t outputBytesHost[BATCH_SIZE];
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
  void* d_out_data[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&d_out_data[i], batch_bytes_host[i]));
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompError_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedLZ4CompressGetTempSize(
      (const void* const*)d_in_data,
      batch_bytes_host,
      BATCH_SIZE,
      &comp_opts,
      &comp_temp_bytes);
  REQUIRE(status == cudaSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t comp_out_bytes[BATCH_SIZE];
  status = nvcompBatchedLZ4CompressGetOutputSize(
      (const void* const*)d_in_data,
      batch_bytes_host,
      BATCH_SIZE,
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      comp_out_bytes);
  REQUIRE(status == cudaSuccess);

  void* d_comp_out[11];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&d_comp_out[i], comp_out_bytes[i]));
  }

  status = nvcompBatchedLZ4CompressAsync(
      (const void* const*)d_in_data,
      batch_bytes_host,
      BATCH_SIZE,
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes,
      stream);
  REQUIRE(status == cudaSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    cudaFree(d_in_data[i]);
  }

  // perform decompression with separate calls
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
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
        output_host[i], out_ptr, output_bytes, cudaMemcpyDeviceToHost));

    cudaFree(temp_ptr);
    cudaFree(d_comp_out[i]);

    // Verify correctness
    for (size_t j = 0; j < batch_sizes_host[i]; ++j) {
      REQUIRE(output_host[i][j] == input_host[i][j]);
    }
    free(output_host[i]);
    free(input_host[i]);
  }

  return 1;

#undef BATCH_SIZE
}

int test_batch_decompression(void)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

#define BATCH_SIZE 4

  const size_t input_size[BATCH_SIZE] = {16, 16, 16, 2};

  const int* input[BATCH_SIZE];

  const int d0[16] = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};
  const int d1[16] = {0, 4, 4, 1, 1, 1, 0, 0, 0, 3, 2, 2, 2, 2, 2, 2};
  const int d2[16] = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};
  const int d3[2] = {1, 2};

  input[0] = d0;
  input[1] = d1;
  input[2] = d2;
  input[3] = d3;

  void* compressed[BATCH_SIZE];
  size_t comp_out_bytes[BATCH_SIZE];

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  {
    // create GPU only input buffer
    void* d_in_data;
    size_t chunk_size = 1 << 16;

    for (int i = 0; i < BATCH_SIZE; i++) {
      size_t in_bytes = sizeof(T) * input_size[i];
      CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));

      CUDA_CHECK(
          cudaMemcpy(d_in_data, input[i], in_bytes, cudaMemcpyHostToDevice));

      nvcompLZ4FormatOpts comp_opts;
      comp_opts.chunk_size = chunk_size;

      nvcompError_t status;

      // Compress on the GPU
      size_t comp_temp_bytes;
      status = nvcompLZ4CompressGetTempSize(
          d_in_data, in_bytes, type, &comp_opts, &comp_temp_bytes);
      REQUIRE(status == nvcompSuccess);

      void* d_comp_temp;
      CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

      status = nvcompLZ4CompressGetOutputSize(
          d_in_data,
          in_bytes,
          type,
          &comp_opts,
          d_comp_temp,
          comp_temp_bytes,
          &comp_out_bytes[i],
          0);
      REQUIRE(status == nvcompSuccess);

      void* d_comp_out;
      CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes[i]));

      status = nvcompLZ4CompressAsync(
          d_in_data,
          in_bytes,
          type,
          &comp_opts,
          d_comp_temp,
          comp_temp_bytes,
          d_comp_out,
          &comp_out_bytes[i],
          stream);
      REQUIRE(status == nvcompSuccess);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      compressed[i] = malloc(comp_out_bytes[i]);

      CUDA_CHECK(cudaMemcpy(
          compressed[i],
          d_comp_out,
          comp_out_bytes[i],
          cudaMemcpyDeviceToHost));

      cudaFree(d_comp_temp);
      cudaFree(d_comp_out);
      cudaFree(d_in_data);
    }
  } // Block to separate compression and decompression

  {

    void* d_compressed[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
      CUDA_CHECK(cudaMalloc(&d_compressed[i], comp_out_bytes[i]));
      cudaMemcpy(
          d_compressed[i],
          compressed[i],
          comp_out_bytes[i],
          cudaMemcpyHostToDevice);
    }

    nvcompError_t status;
    void* metadata_ptr;

    status = nvcompBatchedLZ4DecompressGetMetadata(
        (const void**)d_compressed,
        comp_out_bytes,
        BATCH_SIZE,
        &metadata_ptr,
        stream);

    // LZ4 decompression does not need temp space.
    size_t temp_bytes;
    status = nvcompBatchedLZ4DecompressGetTempSize(
        (const void*)metadata_ptr, &temp_bytes);

    void* temp_ptr;
    CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

    size_t decomp_out_bytes[BATCH_SIZE];
    status = nvcompBatchedLZ4DecompressGetOutputSize(
        (const void*)metadata_ptr, BATCH_SIZE, decomp_out_bytes);

    void* d_decomp_out[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
      CUDA_CHECK(cudaMalloc(&d_decomp_out[i], decomp_out_bytes[i]));
    }

    status = nvcompBatchedLZ4DecompressAsync(
        (const void* const*)d_compressed,
        comp_out_bytes,
        BATCH_SIZE,
        temp_ptr,
        temp_bytes,
        metadata_ptr,
        (void* const*)d_decomp_out,
        decomp_out_bytes,
        stream);

    REQUIRE(status == nvcompSuccess);

    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    nvcompBatchedLZ4DecompressDestroyMetadata(metadata_ptr);
    cudaFree(temp_ptr);

    T* output_host[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
      output_host[i] = malloc(decomp_out_bytes[i]);
      cudaMemcpy(
          output_host[i],
          d_decomp_out[i],
          decomp_out_bytes[i],
          cudaMemcpyDeviceToHost);
      // Verify correctness
      REQUIRE(decomp_out_bytes[i] == input_size[i] * sizeof(T));
      for (size_t j = 0; j < decomp_out_bytes[i] / sizeof(T); ++j) {
        REQUIRE(output_host[i][j] == input[i][j]);
      }
      free(output_host[i]);
    }

    for (int i = 0; i < BATCH_SIZE; i++) {
      cudaFree(d_compressed[i]);
      cudaFree(d_decomp_out[i]);
      free(compressed[i]);
    }
  }
  return 1;

#undef BATCH_SIZE
}

int test_batch_compression_and_decompression(void)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  nvcompLZ4FormatOpts comp_opts = {1 << 16};

  // set a constant seed
  srand(0);

#define BATCH_SIZE 1130

  // prepare input and output on host
  size_t batch_sizes_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_sizes_host[i] = ((i * 1103) % 100000) + 10000;
  }

  size_t batch_bytes_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    batch_bytes_host[i] = sizeof(T) * batch_sizes_host[i];
  }

  T* input_host[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    input_host[i] = malloc(sizeof(T) * batch_sizes_host[i]);
    for (size_t j = 0; j < batch_sizes_host[i]; ++j) {
      // make sure there should be some repeats to compress
      input_host[i][j] = (rand() % 4) + 300;
    }
  }

  size_t outputBytesHost[BATCH_SIZE];
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
  void* d_out_data[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&d_out_data[i], batch_bytes_host[i]));
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompError_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedLZ4CompressGetTempSize(
      (const void* const*)d_in_data,
      batch_bytes_host,
      BATCH_SIZE,
      &comp_opts,
      &comp_temp_bytes);
  REQUIRE(status == cudaSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t comp_out_bytes[BATCH_SIZE];
  status = nvcompBatchedLZ4CompressGetOutputSize(
      (const void* const*)d_in_data,
      batch_bytes_host,
      BATCH_SIZE,
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      comp_out_bytes);
  REQUIRE(status == cudaSuccess);

  void* d_comp_out[BATCH_SIZE];
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    CUDA_CHECK(cudaMalloc(&d_comp_out[i], comp_out_bytes[i]));
  }

  status = nvcompBatchedLZ4CompressAsync(
      (const void* const*)d_in_data,
      batch_bytes_host,
      BATCH_SIZE,
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes,
      stream);
  REQUIRE(status == cudaSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    cudaFree(d_in_data[i]);
  }

  void* metadata_ptr;

  status = nvcompBatchedLZ4DecompressGetMetadata(
      (const void**)d_comp_out,
      comp_out_bytes,
      BATCH_SIZE,
      &metadata_ptr,
      stream);

  // LZ4 decompression does not need temp space.
  size_t temp_bytes;
  status = nvcompBatchedLZ4DecompressGetTempSize(
      (const void*)metadata_ptr, &temp_bytes);

  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

  size_t decomp_out_bytes[BATCH_SIZE];
  status = nvcompBatchedLZ4DecompressGetOutputSize(
      (const void*)metadata_ptr, BATCH_SIZE, decomp_out_bytes);

  void* d_decomp_out[BATCH_SIZE];
  for (int i = 0; i < BATCH_SIZE; i++) {
    CUDA_CHECK(cudaMalloc(&d_decomp_out[i], decomp_out_bytes[i]));
  }

  status = nvcompBatchedLZ4DecompressAsync(
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

  cudaError_t err = cudaDeviceSynchronize();
  REQUIRE(err == cudaSuccess);

  nvcompBatchedLZ4DecompressDestroyMetadata(metadata_ptr);
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
  int num_tests = 3;
  int rv = 0;

  if (!test_batch_compression()) {
    printf("compression only test failed.");
    rv += 1;
  }

  if (!test_batch_decompression()) {
    printf("decompression only test failed.");
    rv += 1;
  }

  if (!test_batch_compression_and_decompression()) {
    printf("compression and decompression test failed.");
    rv += 1;
  }

  if (rv == 0) {
    printf("SUCCESS: All tests passed: %d/%d\n", (num_tests - rv), num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", rv, num_tests);
  }

  return rv;
}
