/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "../src/common.h"
#include "catch.hpp"
#include "lz4.hpp"
#include "nvcomp.hpp"

#include <cascaded.h>
#include <nvcomp.h>
#include <vector>

#include <cuda_profiler_api.h>
#include <iomanip>
#include <random>

using namespace nvcomp;

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw;                                                                   \
    }                                                                          \
  } while (0);


template <typename valT, typename runT>
void random_runs(
    std::vector<valT>& res, const valT max_val, const runT max_run, int seed)
{
  std::mt19937 eng(seed);
  std::uniform_int_distribution<> distr(0, max_run);

  for (valT val = 0; val < max_val; val++) {
    runT run = distr(eng);
    res.insert(res.end(), run, val);
  }
}

template <typename T>
void dump(const std::string desc, std::vector<T>& data, size_t size)
{
#if VERBOSE > 0
  std::cout << desc << ": ";
  for (size_t i = 0; i < size; i++)
    std::cout << data[i] << " ";
  std::cout << std::endl;
#endif
}

// Test method that takes an input data, compresses it (on the CPU),
// decompresses it on the GPU, and verifies it is correct.
// Uses Cascaded Compression
template <typename T>
void test(
    const std::vector<T>& data,
    size_t /*chunk_size*/,
    int numRLEs,
    int numDeltas,
    int bitPacking)
{
  const nvcompType_t type = nvcomp::getnvcompType<T>();

#if VERBOSE > 1
  // dump input data
  std::cout << "Input" << std::endl;
  for (size_t i = 0; i < data.size(); i++)
    std::cout << data[i] << " ";
  std::cout << std::endl;
#endif

  // these two items will be the only forms of communication between
  // compression and decompression
  void* d_comp_out;
  size_t comp_out_bytes;

  {
    // this block handles compression, and we scope it to ensure only
    // serialized metadata and compressed data, are the only things passed
    // between compression and decopmression
    std::cout << "----------" << std::endl;
    std::cout << "uncompressed (B): " << data.size() * sizeof(T) << std::endl;

    // create GPU only input buffer
    void* d_in_data;
    const size_t in_bytes = sizeof(T) * data.size();
    CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_in_data, data.data(), in_bytes, cudaMemcpyHostToDevice));

    nvcompCascadedFormatOpts comp_opts;
    comp_opts.num_RLEs = numRLEs;
    comp_opts.num_deltas = numDeltas;
    comp_opts.use_bp = bitPacking;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvcompError_t status;

    // Compress on the GPU
    size_t comp_temp_bytes;
    status = nvcompCascadedCompressGetTempSize(
        d_in_data,
        in_bytes,
        nvcomp::getnvcompType<T>(),
        &comp_opts,
        &comp_temp_bytes);
    REQUIRE(status == nvcompSuccess);

    void* d_comp_temp;
    CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

    status = nvcompCascadedCompressGetOutputSize(
        d_in_data,
        in_bytes,
        nvcomp::getnvcompType<T>(),
        &comp_opts,
        d_comp_temp,
        comp_temp_bytes,
        &comp_out_bytes,
        false);
    REQUIRE(status == nvcompSuccess);

    CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

    status = nvcompCascadedCompressAsync(
        d_in_data,
        in_bytes,
        nvcomp::getnvcompType<T>(),
        &comp_opts,
        d_comp_temp,
        comp_temp_bytes,
        d_comp_out,
        &comp_out_bytes,
        stream);
    REQUIRE(status == nvcompSuccess);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_comp_temp);
    cudaFree(d_in_data);
    cudaStreamDestroy(stream);

    std::cout << "comp_size: " << comp_out_bytes
              << ", compressed ratio: " << std::fixed << std::setprecision(2)
              << (double)in_bytes / comp_out_bytes << std::endl;
  }

  {
    // this block handles decompression, and we scope it to ensure only
    // serialized metadata and compressed data, are the only things passed
    // between compression and decopmression

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // get metadata from compressed data
    void* metadata;
    nvcompError_t err = nvcompDecompressGetMetadata(
        d_comp_out, comp_out_bytes, &metadata, stream);
    REQUIRE(err == nvcompSuccess);

    // get temp size
    size_t decomp_temp_bytes;
    err = nvcompDecompressGetTempSize(metadata, &decomp_temp_bytes);
    REQUIRE(err == nvcompSuccess);

    // get output buffer size
    size_t decomp_out_bytes;
    err = nvcompDecompressGetOutputSize(metadata, &decomp_out_bytes);
    REQUIRE(err == nvcompSuccess);

    // allocate temp buffer
    void* d_decomp_temp;
    CUDA_CHECK(cudaMalloc(
        &d_decomp_temp, decomp_temp_bytes)); // also can use RMM_ALLOC instead

    // allocate output buffer
    void* decomp_out_ptr;
    CUDA_CHECK(cudaMalloc(
        &decomp_out_ptr, decomp_out_bytes)); // also can use RMM_ALLOC instead

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    nvcompError_t status;

    // execute decompression (asynchronous)
    err = nvcompDecompressAsync(
        d_comp_out,
        comp_out_bytes,
        d_decomp_temp,
        decomp_temp_bytes,
        metadata,
        decomp_out_ptr,
        decomp_out_bytes,
        stream);
    REQUIRE(err == nvcompSuccess);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // stop timing and the profiler
    clock_gettime(CLOCK_MONOTONIC, &end);
    std::cout << "throughput (GB/s): " << gbs(start, end, decomp_out_bytes)
              << std::endl;

    nvcompDecompressDestroyMetadata(metadata);

    cudaStreamDestroy(stream);
    cudaFree(d_decomp_temp);
    cudaFree(d_comp_out);

    //  int* res = (int*)malloc(decomp_bytes);
    std::vector<T> res(decomp_out_bytes / sizeof(T));
    cudaMemcpy(
        &res[0], decomp_out_ptr, decomp_out_bytes, cudaMemcpyDeviceToHost);

#if VERBOSE > 1
    // dump output data
    std::cout << "Output" << std::endl;
    for (size_t i = 0; i < data.size(); i++)
      std::cout << ((T*)out_ptr)[i] << " ";
    std::cout << std::endl;
#endif

    REQUIRE(res == data);
  }
}


