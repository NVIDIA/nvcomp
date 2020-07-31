/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "cascaded.h"
#include "nvcomp.h"

#include "cuda_runtime.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>

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

int test_rle_delta(void)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  int packing = 0;
  int RLE = 1;
  int Delta = 1;

  const size_t inputSize = 16;
  T input[16] = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};

  // create GPU only input buffer
  void* d_in_data;
  const size_t in_bytes = sizeof(T) * inputSize;
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

  nvcompCascadedFormatOpts comp_opts;
  comp_opts.num_RLEs = RLE;
  comp_opts.num_deltas = Delta;
  comp_opts.use_bp = packing;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompError_t status;

  // Compress on the GPU
  size_t comp_temp_bytes;
  status = nvcompCascadedCompressGetTempSize(
      d_in_data, in_bytes, type, &comp_opts, &comp_temp_bytes);
  REQUIRE(status == cudaSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t comp_out_bytes;
  status = nvcompCascadedCompressGetOutputSize(
      d_in_data,
      in_bytes,
      type,
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      &comp_out_bytes,
      0);
  REQUIRE(status == cudaSuccess);

  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  status = nvcompCascadedCompressAsync(
      d_in_data,
      in_bytes,
      type,
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      &comp_out_bytes,
      stream);
  REQUIRE(status == cudaSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  // select compression algorithm
  // Get metadata
  void* metadata_ptr;
  status = nvcompDecompressGetMetadata(
      d_comp_out, comp_out_bytes, &metadata_ptr, stream);
  REQUIRE(status == cudaSuccess);

  // get temp size
  size_t temp_bytes;
  status = nvcompDecompressGetTempSize(metadata_ptr, &temp_bytes);
  REQUIRE(status == cudaSuccess);

  // allocate temp buffer
  void* temp_ptr;
  cudaMalloc(&temp_ptr, temp_bytes); // also can use RMM_ALLOC instead

  // get output size
  size_t output_bytes;
  status = nvcompDecompressGetOutputSize(metadata_ptr, &output_bytes);
  REQUIRE(status == cudaSuccess);

  // allocate output buffer
  void* out_ptr;
  cudaMalloc(&out_ptr, output_bytes); // also can use RMM_ALLOC instead

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
  REQUIRE(status == cudaSuccess);

  status = cudaDeviceSynchronize();
  REQUIRE(status == cudaSuccess);

  nvcompDecompressDestroyMetadata(metadata_ptr);

  // Copy result back to host
  int res[16];
  cudaMemcpy(res, out_ptr, output_bytes, cudaMemcpyDeviceToHost);

  cudaFree(temp_ptr);
  cudaFree(d_comp_out);

  // Verify correctness
  for (size_t i = 0; i < inputSize; ++i) {
    REQUIRE(res[i] == input[i]);
  }

  return 1;
}

int test_rle_delta_bp(void)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  int packing = 1;
  int RLE = 1;
  int Delta = 1;

  const size_t inputSize = 12;
  const int input[12] = {0, 2, 2, 3, 0, 0, 3, 1, 1, 1, 1, 1};

  // create GPU only input buffer
  void* d_in_data;
  const size_t in_bytes = sizeof(T) * inputSize;
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

  nvcompCascadedFormatOpts comp_opts;
  comp_opts.num_RLEs = RLE;
  comp_opts.num_deltas = Delta;
  comp_opts.use_bp = packing;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompError_t status;

  // Compress on the GPU
  size_t comp_temp_bytes;
  status = nvcompCascadedCompressGetTempSize(
      d_in_data, in_bytes, type, &comp_opts, &comp_temp_bytes);
  REQUIRE(status == cudaSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t comp_out_bytes;
  status = nvcompCascadedCompressGetOutputSize(
      d_in_data,
      in_bytes,
      type,
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      &comp_out_bytes,
      0);
  REQUIRE(status == cudaSuccess);

  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  status = nvcompCascadedCompressAsync(
      d_in_data,
      in_bytes,
      type,
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      &comp_out_bytes,
      stream);
  REQUIRE(status == cudaSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  // Get metadata
  void* metadata_ptr;
  status = nvcompDecompressGetMetadata(
      d_comp_out, comp_out_bytes, &metadata_ptr, stream);
  REQUIRE(status == cudaSuccess);

  // get temp size
  size_t temp_bytes;
  status = nvcompDecompressGetTempSize(metadata_ptr, &temp_bytes);
  REQUIRE(status == cudaSuccess);

  // allocate temp buffer
  void* temp_ptr;
  cudaMalloc(&temp_ptr, temp_bytes); // also can use RMM_ALLOC instead

  // get output size
  size_t output_bytes;
  status = nvcompDecompressGetOutputSize(metadata_ptr, &output_bytes);
  REQUIRE(status == cudaSuccess);

  // allocate output buffer
  void* out_ptr;
  cudaMalloc(&out_ptr, output_bytes); // also can use RMM_ALLOC instead

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
  REQUIRE(status == cudaSuccess);

  status = cudaStreamSynchronize(stream);
  REQUIRE(status == cudaSuccess);

  // Destory the metadata object and free memory
  nvcompDecompressDestroyMetadata(metadata_ptr);

  // Copy result back to host
  int res[12];
  cudaMemcpy(res, out_ptr, output_bytes, cudaMemcpyDeviceToHost);

  cudaFree(temp_ptr);
  cudaFree(d_comp_out);

  // Verify result
  for (size_t i = 0; i < inputSize; ++i) {
    REQUIRE(res[i] == input[i]);
  }
}

int main(int argc, char** argv)
{
  int num_tests = 2;
  int rv = 0;

  if (!test_rle_delta()) {
    printf("rle_delta test failed.");
    rv += 1;
  }

  if (!test_rle_delta_bp()) {
    printf("rle_delta_bp test failed.");
    rv += 1;
  }

  if (rv == 0) {
    printf("SUCCESS: All tests passed: %d/%d\n", (num_tests - rv), num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", rv, num_tests);
  }

  return rv;
}
