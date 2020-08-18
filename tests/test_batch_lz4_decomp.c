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

#include "lz4.h"
#include "nvcomp.h"

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

#define NUM_CHUNKS 4

int test_bulk_decomp(void)
{
  const typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

//  const int NUM_CHUNKS=4;

  const size_t inputSize[NUM_CHUNKS] = {16, 16, 16, 2};

  int* input[NUM_CHUNKS];
  
  int d0[16] = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};
  int d1[16] = {0, 4, 4, 1, 1, 1, 0, 0, 0, 3, 2, 2, 2, 2, 2, 2};
  int d2[16] = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};
  int d3[2] = {1, 2};

  input[0] = d0;
  input[1] = d1;
  input[2] = d2;
  input[3] = d3;

  void* compressed[NUM_CHUNKS];
  size_t comp_out_bytes[NUM_CHUNKS];

  cudaStream_t stream;
  cudaStreamCreate(&stream);

{
  // create GPU only input buffer
  void* d_in_data;
  size_t chunk_size = 1 << 16;

  for(int i=0; i<NUM_CHUNKS; i++) {

    size_t in_bytes = sizeof(T) * inputSize[i];
    CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));

    CUDA_CHECK(cudaMemcpy(d_in_data, input[i], in_bytes, cudaMemcpyHostToDevice));

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

    CUDA_CHECK(cudaMemcpy(compressed[i], d_comp_out, comp_out_bytes[i], cudaMemcpyDeviceToHost));

    cudaFree(d_comp_temp);
    cudaFree(d_comp_out);
    cudaFree(d_in_data);
  }
} // Block to separate compression and decompression  

{

  void* d_compressed[NUM_CHUNKS];
  for(int i=0; i<NUM_CHUNKS; i++) {
    CUDA_CHECK(cudaMalloc(&d_compressed[i], comp_out_bytes[i]));
    cudaMemcpy(d_compressed[i], compressed[i], comp_out_bytes[i], cudaMemcpyHostToDevice);
  }

  nvcompError_t status;
  void* metadata_ptr;
  
  status = nvcompBatchedLZ4DecompressGetMetadata(
      (const void**)d_compressed,
      comp_out_bytes,
      NUM_CHUNKS,
      &metadata_ptr,
      stream);


// LZ4 decompression does not need temp space.
  size_t temp_bytes;
  status = nvcompBatchedLZ4DecompressGetTempSize((const void*)metadata_ptr, &temp_bytes);

  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));
  
  size_t decomp_out_bytes[NUM_CHUNKS];
  status = nvcompBatchedLZ4DecompressGetOutputSize((const void*)metadata_ptr, NUM_CHUNKS, decomp_out_bytes);

  void* d_decomp_out[NUM_CHUNKS];
  for(int i=0; i<NUM_CHUNKS; i++) {
    CUDA_CHECK(cudaMalloc(&d_decomp_out[i], decomp_out_bytes[i]));
  }

  status = nvcompBatchedLZ4DecompressAsync(
      (const void* const*)d_compressed,
      comp_out_bytes,
      NUM_CHUNKS,
      temp_ptr,
      temp_bytes,
      (const void*)metadata_ptr,
      (void* const*)d_decomp_out,
      decomp_out_bytes,
      stream);

  REQUIRE(status == nvcompSuccess);

  cudaError_t err = cudaDeviceSynchronize();
  REQUIRE(err == cudaSuccess);

  nvcompBatchedLZ4DecompressDestroyMetadata(metadata_ptr);
  cudaFree(temp_ptr);

  T res[NUM_CHUNKS][inputSize[0]];
  for(int i=0; i<NUM_CHUNKS; i++) {
    cudaMemcpy((void*)res[i], d_decomp_out[i], decomp_out_bytes[i], cudaMemcpyDeviceToHost);
    // Verify correctness
    for (size_t j = 0; j < decomp_out_bytes[i]/sizeof(T); ++j) {
      REQUIRE(res[i][j] == input[i][j]);
    }
  }

  for(int i=0; i<NUM_CHUNKS; i++) {
    cudaFree(d_compressed[i]);
    free(compressed[i]);
    cudaFree(d_decomp_out[i]);
  }
    

}
  return 1;
}

int main()
{
  int num_tests = 2;
  int rv = 0;

  if (!test_bulk_decomp()) {
    printf("bull_decomp test failed.");
    rv += 1;
  }

  if (rv == 0) {
    printf("SUCCESS: All tests passed: %d/%d\n", (num_tests - rv), num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", rv, num_tests);
  }

  return rv;
}
