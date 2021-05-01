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

#define CATCH_CONFIG_MAIN

#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

#include "catch.hpp"
#include <assert.h>
#include <stdlib.h>
#include <vector>

// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace nvcomp;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

TEST_CASE("comp/decomp RLE-Delta", "[nvcomp]")
{
  using T = int;

  int packing = 0;
  int RLE = 1;
  int Delta = 1;
  std::vector<T> input = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};
  size_t chunk_size = 10000;

  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t comp_temp_bytes = 0;
  size_t comp_out_bytes = 0;
  void* d_comp_temp;
  void* d_comp_out;

  // Get comptess temp size
  CascadedCompressor compressor(getnvcompType<T>(), RLE, Delta, packing);

  compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);
  REQUIRE(comp_temp_bytes > 0);
  REQUIRE(comp_out_bytes > 0);

  // allocate temp buffer
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Allocate output buffer
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  size_t* comp_out_bytes_ptr;
  cudaMalloc((void**)&comp_out_bytes_ptr, sizeof(*comp_out_bytes_ptr));
  compressor.compress_async(
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes_ptr,
      stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(
      &comp_out_bytes,
      comp_out_bytes_ptr,
      sizeof(comp_out_bytes),
      cudaMemcpyDeviceToHost));
  cudaFree(comp_out_bytes_ptr);

  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  size_t temp_bytes = 0;
  size_t num_out_bytes = 0;
  void* temp_ptr;
  T* out_ptr;

  CascadedDecompressor decompressor;

  // get temp size
  decompressor.configure(
      d_comp_out, comp_out_bytes, &temp_bytes, &num_out_bytes, stream);
  REQUIRE(temp_bytes > 0);
  REQUIRE(num_out_bytes == in_bytes);

  // allocate temp buffer
  cudaMalloc(&temp_ptr, temp_bytes); // also can use RMM_ALLOC instead

  // allocate output buffer
  cudaMalloc(&out_ptr, num_out_bytes); // also can use RMM_ALLOC instead

  // execute decompression (asynchronous)
  decompressor.decompress_async(
      d_comp_out,
      comp_out_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      num_out_bytes,
      stream);

  cudaStreamSynchronize(stream);

  // Copy result back to host
  std::vector<T> res(num_out_bytes / sizeof(T));
  cudaMemcpy(&res[0], out_ptr, num_out_bytes, cudaMemcpyDeviceToHost);

  cudaFree(temp_ptr);
  cudaFree(d_comp_out);
  cudaFree(out_ptr);

  // Verify correctness
  REQUIRE(res == input);
}

TEST_CASE("comp/decomp RLE-Delta-BP", "[nvcomp]")
{
  using T = int;

  int packing = 1;
  int RLE = 1;
  int Delta = 1;
  std::vector<T> input = {0, 2, 2, 3, 0, 0, 3, 1, 1, 1, 1, 1};
  size_t chunk_size = 10000;

  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t comp_temp_bytes = 0;
  size_t comp_out_bytes = 0;
  void* d_comp_temp;
  void* d_comp_out;

  CascadedCompressor compressor(getnvcompType<T>(), RLE, Delta, packing);

  compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);
  REQUIRE(comp_temp_bytes > 0);
  REQUIRE(comp_out_bytes > 0);

  // allocate temp buffer
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Allocate output buffer
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  size_t* comp_out_bytes_ptr;
  cudaMallocHost((void**)&comp_out_bytes_ptr, sizeof(*comp_out_bytes_ptr));

  compressor.compress_async(
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes_ptr,
      stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  comp_out_bytes = *comp_out_bytes_ptr;

  cudaFreeHost(comp_out_bytes_ptr);
  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  size_t temp_bytes;
  void* temp_ptr;
  size_t num_out_bytes = 0;
  T* out_ptr = nullptr;

  CascadedDecompressor decompressor;
  decompressor.configure(
      d_comp_out, comp_out_bytes, &temp_bytes, &num_out_bytes, stream);

  // allocate temp buffer
  cudaMalloc(&temp_ptr, temp_bytes); // also can use RMM_ALLOC instead

  // allocate output buffer
  cudaMalloc(&out_ptr, num_out_bytes); // also can use RMM_ALLOC instead

  // execute decompression (asynchronous)
  decompressor.decompress_async(
      d_comp_out,
      comp_out_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      num_out_bytes,
      stream);

  cudaStreamSynchronize(stream);

  // Copy result back to host
  std::vector<T> res(num_out_bytes / sizeof(T));
  cudaMemcpy(&res[0], out_ptr, num_out_bytes, cudaMemcpyDeviceToHost);
  cudaFree(out_ptr);

  // Verify result
  REQUIRE(res == input);
  cudaFree(temp_ptr);
  cudaFree(d_comp_out);
}

TEST_CASE("max_size_test", "[nvcomp]")
{
  using T = uint8_t;

  int packing = 1;
  int RLE = 1;
  int Delta = 1;
  const size_t size = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  std::vector<T> input(size);

  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t comp_temp_bytes = 0;
  size_t comp_out_bytes = 0;
  void* d_comp_temp = nullptr;
  void* d_comp_out = nullptr;

  size_t* comp_out_bytes_ptr;
  cudaMallocHost((void**)&comp_out_bytes_ptr, sizeof(*comp_out_bytes_ptr));

  try {
    CascadedCompressor compressor(getnvcompType<T>(), RLE, Delta, packing);

    compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);

    // allocate temp buffer
    CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

    compressor.compress_async(
        d_in_data,
        in_bytes,
        d_comp_temp,
        comp_temp_bytes,
        d_comp_out,
        comp_out_bytes_ptr,
        stream);

    // should have thrown an exception by now
    REQUIRE(false);
  } catch (const NVCompException&) {
    // we through the right exception, pass
  }

  cudaFreeHost(comp_out_bytes_ptr);

  if (d_comp_temp) {
    cudaFree(d_comp_temp);
  }
  if (d_comp_out) {
    cudaFree(d_comp_out);
  }
}
