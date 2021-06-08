/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp.h"
#include "nvcomp/cascaded.h"
#include "nvcomp/lz4.h"

#include "cuda_runtime.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Test GPU generic decompression with cascaded and lz4 compression APIs

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

int check_decompress(
    const int* const input,
    size_t input_size,
    void* d_comp_out,
    size_t comp_out_bytes,
    cudaStream_t stream)
{
  // get temp and output size
  size_t temp_bytes;
  size_t output_bytes;
  void* metadata_ptr = NULL;

  nvcompError_t status = nvcompDecompressGetMetadata(
      d_comp_out, comp_out_bytes, &metadata_ptr, stream);
  REQUIRE(status == nvcompSuccess);

  status = nvcompDecompressGetTempSize(metadata_ptr, &temp_bytes);
  REQUIRE(status == nvcompSuccess);

  // allocate temp buffer
  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

  status = nvcompDecompressGetOutputSize(metadata_ptr, &output_bytes);
  REQUIRE(status == nvcompSuccess);

  // allocate output buffer
  void* out_ptr;
  CUDA_CHECK(cudaMalloc(&out_ptr, output_bytes));

  // execute decompression (asynchronous)
  status = nvcompDecompressAsync(
      d_comp_out,
      comp_out_bytes,
      temp_ptr,
      temp_bytes,
      metadata_ptr,
      out_ptr,
      output_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  nvcompCascadedDestroyMetadata(metadata_ptr);

  // Copy result back to host
  int* result = malloc(input_size * sizeof(int));
  CUDA_CHECK(cudaMemcpy(result, out_ptr, output_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(temp_ptr));

  // Verify correctness
  for (size_t i = 0; i < input_size; ++i) {
    REQUIRE(result[i] == input[i]);
  }
  free(result);

  return 1;
}

int test_cascaded(void)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  int packing = 0;
  int RLE = 1;
  int Delta = 1;

  const size_t input_size = 16;
  T input[16] = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};

  // create GPU only input buffer
  void* d_in_data;
  const size_t in_bytes = sizeof(T) * input_size;
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

  nvcompCascadedFormatOpts comp_opts;
  comp_opts.num_RLEs = RLE;
  comp_opts.num_deltas = Delta;
  comp_opts.use_bp = packing;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcompError_t status;

  // Compress on the GPU
  size_t comp_temp_bytes;
  size_t comp_out_bytes;
  size_t metadata_bytes;
  status = nvcompCascadedCompressConfigure(
      &comp_opts,
      type,
      in_bytes,
      &metadata_bytes,
      &comp_temp_bytes,
      &comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  size_t* d_comp_out_bytes;
  CUDA_CHECK(cudaMalloc((void**)&d_comp_out_bytes, sizeof(*d_comp_out_bytes)));
  status = nvcompCascadedCompressAsync(
      &comp_opts,
      type,
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      d_comp_out_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(
      &comp_out_bytes,
      d_comp_out_bytes,
      sizeof(comp_out_bytes),
      cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_comp_out_bytes));
  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaFree(d_in_data));

  int rv
      = check_decompress(input, input_size, d_comp_out, comp_out_bytes, stream);
  CUDA_CHECK(cudaFree(d_comp_out));

  return rv;
}

int test_lz4(void)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  const size_t input_size = 123000;

  T* input = malloc(input_size * sizeof(T));
  for (size_t i = 0; i < input_size; ++i) {
    // semi compressible data
    input[i] = (i % 23) + (i / 101);
  }

  // create GPU only input buffer
  void* d_in_data;
  const size_t in_bytes = sizeof(T) * input_size;
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

  nvcompLZ4FormatOpts opts;
  opts.chunk_size = 1 << 16;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcompError_t status;

  size_t* p_comp_out_bytes;
  CUDA_CHECK(
      cudaMallocHost((void**)&p_comp_out_bytes, sizeof(*p_comp_out_bytes)));

  // Compress on the GPU
  size_t comp_temp_bytes;
  size_t metadata_bytes;
  status = nvcompLZ4CompressConfigure(
      &opts,
      type,
      in_bytes,
      &metadata_bytes,
      &comp_temp_bytes,
      p_comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
  CUDA_CHECK(cudaMalloc(&d_comp_out, *p_comp_out_bytes));

  status = nvcompLZ4CompressAsync(
      &opts,
      type,
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      p_comp_out_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  int rv = check_decompress(
      input, input_size, d_comp_out, *p_comp_out_bytes, stream);
  CUDA_CHECK(cudaFree(d_comp_out));
  CUDA_CHECK(cudaFreeHost(p_comp_out_bytes));

  free(input);

  return rv;
}

int main(int argc, char** argv)
{
  if (argc != 1) {
    printf("ERROR: %s accepts no arguments.\n", argv[0]);
    return 1;
  }

  int num_tests = 2;
  int rv = 0;

  if (!test_cascaded()) {
    printf("cascaded test failed.\n");
    rv += 1;
  }

  if (!test_lz4()) {
    printf("lz4 test failed.\n");
    rv += 1;
  }

  if (rv == 0) {
    printf("SUCCESS: All tests passed: %d/%d\n", (num_tests - rv), num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", rv, num_tests);
  }

  return rv;
}
