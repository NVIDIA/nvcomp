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

#ifndef VERBOSE
#define VERBOSE 0
#endif

#include "benchmark_common.h"
#include "lz4.hpp"

#include <getopt.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


#include <string>

using namespace nvcomp;

// Benchmark performance from the binary data file fname
static void run_benchmark(const std::vector<std::vector<char>>& data)
{
  size_t total_bytes = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  std::vector<uint8_t*> d_in_data(data.size());
  std::vector<size_t> item_bytes(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    item_bytes[i] = data[i].size();
    CUDA_CHECK(cudaMalloc((void**)&d_in_data[i], data[i].size()));
    CUDA_CHECK(
        cudaMemcpy(d_in_data[i], data[i].data(), data[i].size(), cudaMemcpyHostToDevice));
  }




  // compression
  nvcompLZ4FormatOpts comp_opts = {1 << 16};
  nvcompError_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedLZ4CompressGetTempSize(
      (const void* const*)d_in_data.data(),
      item_bytes.data(),
      data.size(),
      &comp_opts,
      &comp_temp_bytes);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  std::vector<size_t> comp_out_bytes(data.size());
  status = nvcompBatchedLZ4CompressGetOutputSize(
      (const void* const*)d_in_data.data(),
      item_bytes.data(),
      data.size(),
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      comp_out_bytes.data());

  std::vector<void*> d_comp_out(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    CUDA_CHECK(cudaMalloc(&d_comp_out[i], comp_out_bytes[i]));
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  status = nvcompBatchedLZ4CompressAsync(
      (const void* const*)d_in_data.data(),
      item_bytes.data(),
      data.size(),
      &comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      (void* const *)d_comp_out.data(),
      comp_out_bytes.data(),
      stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  clock_gettime(CLOCK_MONOTONIC, &end);
  cudaFree(d_comp_temp);
  for (size_t i = 0; i < data.size(); ++i) {
    cudaFree(d_in_data[i]);
  }

  std::cout << "compression throughput (GB/s): "
            << gbs(start, end, total_bytes) << std::endl;

  // LZ4 decompression
  void* metadata_ptr;

  status = nvcompBatchedLZ4DecompressGetMetadata(
      (const void**)d_comp_out.data(),
      comp_out_bytes.data(),
      data.size(),
      &metadata_ptr,
      stream);

  size_t temp_bytes;
  status = nvcompBatchedLZ4DecompressGetTempSize(
      (const void*)metadata_ptr, &temp_bytes);

  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

  std::vector<size_t> decomp_out_bytes(data.size());
  status = nvcompBatchedLZ4DecompressGetOutputSize(
      (const void*)metadata_ptr, data.size(), decomp_out_bytes.data());

  std::vector<void*> d_decomp_out(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    CUDA_CHECK(cudaMalloc(&d_decomp_out[i], decomp_out_bytes[i]));
  }

  clock_gettime(CLOCK_MONOTONIC, &start);

  status = nvcompBatchedLZ4DecompressAsync(
      (const void* const*)d_comp_out.data(),
      comp_out_bytes.data(),
      data.size(),
      temp_ptr,
      temp_bytes,
      metadata_ptr,
      (void* const*)d_decomp_out.data(),
      decomp_out_bytes.data(),
      stream);

  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &end);

  std::cout << "decompression throughput (GB/s): "
            << gbs(start, end, total_bytes) << std::endl;

  nvcompBatchedLZ4DecompressDestroyMetadata(metadata_ptr);
  cudaFree(temp_ptr);

  for (size_t i = 0; i < data.size(); i++) {
    cudaFree(d_comp_out[i]);
    cudaFree(d_decomp_out[i]);
  }

  cudaStreamDestroy(stream);
  (void)status;
}

std::vector<char> readFile(const std::string& filename)
{
  std::vector<char> buffer(4096);
  std::vector<char> host_data;

  std::ifstream fin(filename, std::ifstream::binary);
  fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  size_t num;
  do {
    num = fin.readsome(buffer.data(), buffer.size());
    host_data.insert(host_data.end(), buffer.begin(), buffer.begin() + num);
  } while (num > 0);

  return host_data;
}

std::vector<std::vector<char>>
multi_file(const std::vector<std::string>& filenames)
{
  std::vector<std::vector<char>> split_data;

  for (auto const& filename : filenames) {
    split_data.emplace_back(readFile(filename));
  }

  return split_data;
}



int main(int argc, char* argv[])
{
  std::vector<std::string> file_names(argc-1);

  for (int i = 1; i < argc; ++i) {
    file_names[i-1] = argv[i];
  }

  run_benchmark(multi_file(file_names));

  return 0;
}
