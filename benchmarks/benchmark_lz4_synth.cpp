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
#include <random>
#include <string>
#include <vector>

using namespace nvcomp;

namespace
{

constexpr const size_t CHUNK_SIZE = 1 << 16;

void print_usage()
{
  printf("Usage: benchmark_binary [OPTIONS]\n");
  printf("  %-35s GPU device number (default 0)\n", "-g, --gpu");
  exit(1);
}

// Benchmark performance from the binary data file fname
void run_benchmark(const std::vector<uint8_t>& data)
{
  const size_t num_bytes = data.size();

  // Make sure dataset fits on GPU to benchmark total compression
  size_t freeMem;
  size_t totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  if (freeMem < num_bytes) {
    std::cout << "Insufficient GPU memory to perform compression." << std::endl;
    exit(1);
  }

  const size_t num_chunks = roundUpDiv(num_bytes, CHUNK_SIZE);

  std::cout << "----------" << std::endl;
  std::cout << "uncompressed (B): " << num_bytes << std::endl;
  std::cout << "chunks " << num_chunks << std::endl;

  uint8_t* d_in_data;
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, num_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, data.data(), num_bytes, cudaMemcpyHostToDevice));

  LZ4Compressor<uint8_t> compressor(d_in_data, num_bytes, CHUNK_SIZE);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Get temp size needed for compression
  const size_t comp_temp_bytes = compressor.get_temp_size();

  // Allocate temp workspace
  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t comp_out_bytes
      = compressor.get_max_output_size(d_comp_temp, comp_temp_bytes);
  benchmark_assert(
      comp_out_bytes > 0, "Output size must be greater than zero.");

  // Allocate compressed output buffer
  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // Launch compression
  compressor.compress_async(
      d_comp_temp, comp_temp_bytes, d_comp_out, &comp_out_bytes, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  clock_gettime(CLOCK_MONOTONIC, &end);

  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  std::cout << "comp_size: " << comp_out_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)num_bytes / comp_out_bytes << std::endl;
  std::cout << "compression throughput (GB/s): " << gbs(start, end, num_bytes)
            << std::endl;

  // get metadata from compressed data on GPU
  Decompressor<uint8_t> decompressor(d_comp_out, comp_out_bytes, stream);

  // allocate temp buffer
  const size_t decomp_temp_bytes = decompressor.get_temp_size();
  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(
      &d_decomp_temp, decomp_temp_bytes)); // also can use RMM_ALLOC instead

  // get output size
  const size_t decomp_bytes = decompressor.get_output_size();

  size_t free, total;
  cudaMemGetInfo(&free, &total);

  // allocate output buffer
  uint8_t* decomp_out_ptr;
  CUDA_CHECK(cudaMalloc(
      (void**)&decomp_out_ptr, decomp_bytes)); // also can use RMM_ALLOC instead

  clock_gettime(CLOCK_MONOTONIC, &start);

  // execute decompression (asynchronous)
  decompressor.decompress_async(
      d_decomp_temp, decomp_temp_bytes, decomp_out_ptr, decomp_bytes, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // stop timing and the profiler
  clock_gettime(CLOCK_MONOTONIC, &end);
  std::cout << "decompression throughput (GB/s): "
            << gbs(start, end, decomp_bytes) << std::endl;

  cudaStreamDestroy(stream);
  cudaFree(d_decomp_temp);
  cudaFree(d_comp_out);

  benchmark_assert(
      decomp_bytes == num_bytes, "Decompressed result incorrect size.");

  std::vector<uint8_t> res(num_bytes);
  cudaMemcpy(res.data(), decomp_out_ptr, num_bytes, cudaMemcpyDeviceToHost);

  benchmark_assert(res == data, "Decompressed data does not match input.");
}

std::vector<uint8_t>
gen_data(int max_byte, const size_t size, std::mt19937& rng)
{
  std::uniform_int_distribution<uint8_t> dist(0, max_byte);

  std::vector<uint8_t> data;

  for (size_t i = 0; i < size; ++i) {
    data.emplace_back(dist(rng));
  }

  return data;
}

void run_tests(std::mt19937& rng)
{
  // test all zeros
  for (size_t b = 0; b < 14; ++b) {
    run_benchmark(gen_data(0, CHUNK_SIZE << b, rng));
  }

  // test random bytes
  for (size_t b = 0; b < 14; ++b) {
    run_benchmark(gen_data(255, CHUNK_SIZE << b, rng));
  }
}

} // namespace

int main(int argc, char* argv[])
{
  int gpu_num = 0;

  // Parse command-line arguments
  while (1) {
    int option_index = 0;
    static struct option long_options[]{{"gpu", required_argument, 0, 'g'},
                                        {"help", no_argument, 0, '?'}};
    int c;
    opterr = 0;
    c = getopt_long(argc, argv, "g?", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 'g':
      gpu_num = atoi(optarg);
      break;
    case '?':
    default:
      print_usage();
      exit(1);
    }
  }

  cudaSetDevice(gpu_num);

  std::mt19937 rng(0);

  run_tests(rng);

  return 0;
}
