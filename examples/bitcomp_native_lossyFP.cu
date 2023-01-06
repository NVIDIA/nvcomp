/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

// Simple example to show how to use bitcomp's native lossy API to compress
// floating point data.
//
// Bitcomp's lossy compression performs an on-the-fly integer quantization
// and compresses the resulting integral values with the lossless encoder.
// A smaller delta used for the quantization will typically lower the
// compression ratio, but will increase precision.

#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <native/bitcomp.h>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw;                                                                   \
    }                                                                          \
  } while (0);

#define BITCOMP_CHECK(call)                                                    \
  {                                                                            \
    bitcompResult_t err = call;                                                \
    if (BITCOMP_SUCCESS != err) {                                              \
      fprintf(                                                                 \
          stderr,                                                              \
          "Bitcomp error %d in file '%s' in line %i.\n",                       \
          err,                                                                 \
          __FILE__,                                                            \
          __LINE__);                                                           \
      fflush(stderr);                                                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Kernel to initialize the input data with a sine function
__global__ void initialize(float* input, float dw, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    input[i] = sinf(i * dw);
}

int main()
{
  constexpr size_t n = 1048576;
  constexpr size_t size = n * sizeof(float);

  // Delta used for the integer quantization.
  // The tuning knob between compression ratio and quality
  const float delta = 0.0001f;
  printf("Using delta = %f\n", delta);

  // Let's execute all the GPU code in a non-default stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Allocate and initialize the floating point input
  // Using managed memory to easily check the results.
  float* input;
  CUDA_CHECK(cudaMallocManaged((void**)&input, size));
  constexpr float dw = 2.0f * M_PI / n;
  initialize<<<(n + 1023) / 1024, 1024, 0, stream>>>(input, dw, n);

  // Allocate a buffer for the decompressed data
  float* output;
  CUDA_CHECK(cudaMallocManaged(&output, size));

  // Create a bitcomp plan to compress FP32 data using a signed integer
  // quantization, since the input data contains positive and negative values.
  bitcompHandle_t plan;
  BITCOMP_CHECK(bitcompCreatePlan(
      &plan,                      // Bitcomp handle
      size,                       // Size in bytes of the uncompressed data
      BITCOMP_FP32_DATA,          // Data type
      BITCOMP_LOSSY_FP_TO_SIGNED, // Compression type
      BITCOMP_DEFAULT_ALGO));     // Bitcomp algo, default or sparse

  // Query the maximum size of the compressed data (worst case scenario)
  // and allocate the compressed buffer
  size_t maxlen = bitcompMaxBuflen(size);
  void* compbuf;
  CUDA_CHECK(cudaMalloc(&compbuf, maxlen));

  // Associate the bitcomp plan to the stream, otherwise the compression
  // or decompression would happen in the default stream
  BITCOMP_CHECK(bitcompSetStream(plan, stream));

  // Compress the input data with the chosen quantization delta
  BITCOMP_CHECK(bitcompCompressLossy_fp32(plan, input, compbuf, delta));

  // Wait for the compression kernel to finish
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Query the compressed size
  size_t compsize;
  BITCOMP_CHECK(bitcompGetCompressedSize(compbuf, &compsize));
  float ratio = static_cast<float>(size) / static_cast<float>(compsize);
  printf("Compression ratio = %.1f\n", ratio);

  // Decompress the data
  BITCOMP_CHECK(bitcompUncompress(plan, compbuf, output));

  // Wait for the decompression to finish
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Compare the results
  float maxdiff = 0.0f;
  for (size_t i = 0; i < n; i++)
    maxdiff = std::max(maxdiff, fabsf(output[i] - input[i]));
  printf("Max absolute difference  = %f\n", maxdiff);

  // Clean up
  BITCOMP_CHECK(bitcompDestroyPlan(plan));
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(compbuf));
  CUDA_CHECK(cudaFree(output));

  return 0;
}
