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

#include <algorithm>
#include <chrono>
#include <string.h>

using namespace nvcomp;

static void print_usage()
{
  printf("Usage: benchmark_cascaded_auto [OPTIONS]\n");
  printf("  %-35s Binary dataset filename (required).\n", "-f, --filename");
  printf("  %-35s Datatype (int or long, default int)\n", "-t, --type");
  printf("  %-35s Elements to compress (default entire file)\n", "-z, --size");
  printf("  %-35s GPU device number (default 0)\n", "-g, --gpu");
  printf(
      "  %-35s Output GPU memory allocation sizes (default off)\n",
      "-m --memory");
  exit(1);
}

// Benchmark performance from the binary data file fname
template <typename T>
static void run_benchmark(
    char* fname,
    size_t input_elts,
    int verbose_memory)
{

  std::vector<T> data;
  data = load_dataset_from_binary<T>(fname, &input_elts);

  // Make sure dataset fits on GPU to benchmark total compression
  size_t freeMem;
  size_t totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  if (freeMem < input_elts * sizeof(T)) {
    std::cout << "Insufficient GPU memory to perform compression." << std::endl;
    exit(1);
  }

  std::cout << "----------" << std::endl;
  std::cout << "uncompressed (B): " << data.size() * sizeof(T) << std::endl;

  void* d_in_data;
  const size_t in_bytes = sizeof(T) * input_elts;
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, data.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompError_t status;

  // Get temp size needed for compression
  size_t metadata_bytes;
  size_t comp_temp_bytes;
  size_t comp_out_bytes;

  status = nvcompCascadedCompressConfigure(
      NULL, getnvcompType<T>(), in_bytes, &metadata_bytes, &comp_temp_bytes, &comp_out_bytes);
  benchmark_assert(status == nvcompSuccess, "CompressConfigure not successful");

  // Allocate temp workspace
  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Allocate compressed output buffer
  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  if (verbose_memory) {
    std::cout << "compression memory (input+output+temp) (B): "
              << (in_bytes + comp_out_bytes + comp_temp_bytes) << std::endl;
    std::cout << "compression temp space (B): " << comp_temp_bytes << std::endl;
    std::cout << "compression output space (B): " << comp_out_bytes
              << std::endl;
  }
  
  auto start = std::chrono::steady_clock::now();

  // Launch compression
  status = nvcompCascadedCompressAsync(
      NULL, // Null format_opts causes Selector to run
      getnvcompType<T>(),
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      &comp_out_bytes,
      stream);

  benchmark_assert(
      status == nvcompSuccess, "nvcompCascadedCompressAuto not successful");
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto end = std::chrono::steady_clock::now();

  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  std::cout << "comp_size: " << comp_out_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)data.size() * sizeof(T) / comp_out_bytes << std::endl;
  std::cout << "compression throughput (GB/s): "
            << gbs(start, end, data.size() * sizeof(T)) << std::endl;

  void* metadata_ptr;
  cudaMallocHost(&metadata_ptr, metadata_bytes);

  // get metadata from compressed data on GPU
  status = nvcompCascadedQueryMetadataAsync(
      d_comp_out, comp_out_bytes, metadata_ptr, metadata_bytes, stream);
  benchmark_assert(status == nvcompSuccess, "Failed to get metadata");

  // get temp size
  size_t decomp_temp_bytes;
  size_t decomp_bytes;
  status = nvcompCascadedDecompressConfigure(
      d_comp_out,
      comp_out_bytes,
      &metadata_ptr,
      &metadata_bytes,
      &decomp_temp_bytes,
      &decomp_bytes,
      stream);

  benchmark_assert(status == nvcompSuccess, "Failed to get metadata");


  if (verbose_memory) {
    std::cout << "decompression memory (input+output+temp) (B): "
              << (decomp_bytes + comp_out_bytes + decomp_temp_bytes)
              << std::endl;
    std::cout << "decompression temp space (B): " << decomp_temp_bytes
              << std::endl;
  }

  // allocate temp buffer
  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(
      &d_decomp_temp, decomp_temp_bytes)); // also can use RMM_ALLOC instead

  // allocate output buffer
  void* decomp_out_ptr;
  CUDA_CHECK(cudaMalloc(
      &decomp_out_ptr, decomp_bytes)); // also can use RMM_ALLOC instead

  start = std::chrono::steady_clock::now();

  // execute decompression (asynchronous)
  status = nvcompCascadedDecompressAsync(
      d_comp_out,
      comp_out_bytes,
      metadata_ptr,
      metadata_bytes,
      d_decomp_temp,
      decomp_temp_bytes,
      decomp_out_ptr,
      decomp_bytes,
      stream);
  benchmark_assert(status == nvcompSuccess, "Failed to launch decompress.");

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // stop timing and the profiler
  end = std::chrono::steady_clock::now();
  std::cout << "decompression throughput (GB/s): "
            << gbs(start, end, decomp_bytes) << std::endl;

  nvcompDecompressDestroyMetadata(metadata_ptr);
  
  cudaFree(metadata_ptr);
  cudaStreamDestroy(stream);
  cudaFree(d_decomp_temp);
  cudaFree(d_comp_out);

  benchmark_assert(
      decomp_bytes == input_elts * sizeof(T),
      "Decompressed result incorrect size.");

  std::vector<T> res(input_elts);
  cudaMemcpy(
      (void*)&res[0],
      decomp_out_ptr,
      input_elts * sizeof(T),
      cudaMemcpyDeviceToHost);

  // check the size

#if VERBOSE > 1
  // dump output data
  std::cout << "Output" << std::endl;
  for (size_t i = 0; i < data.size(); i++)
    std::cout << ((T*)out_ptr)[i] << " ";
  std::cout << std::endl;
#endif

  benchmark_assert(res == data, "Decompressed data does not match input.");
}

int main(int argc, char* argv[])
{
  char* fname = NULL;
  int gpu_num = 0;
  int verbose_memory = 0;
  std::string dtype = "int";
  size_t size = 0;

  // Parse command-line arguments
  char** argv_end = argv + argc;
  argv += 1;
  while (argv != argv_end) {
    char* arg = *argv++;
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-?") == 0) {
      print_usage();
      return 1;
    }
    if (strcmp(arg, "--memory") == 0 || strcmp(arg, "-m") == 0) {
      verbose_memory = 1;
      continue;
    }

    // all arguments below require at least a second value in argv
    if (argv >= argv_end) {
      print_usage();
      return 1;
    }

    char* optarg = *argv++;
    if (strcmp(arg, "--filename") == 0 || strcmp(arg, "-f") == 0) {
      fname = optarg;
      continue;
    }
    if (strcmp(arg, "--gpu") == 0 || strcmp(arg, "-g") == 0) {
      gpu_num = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--type") == 0 || strcmp(arg, "-t") == 0) {
      dtype = optarg;
      continue;
    }
    if (strcmp(arg, "--size") == 0 || strcmp(arg, "-gpu") == 0) {
      size = atoll(optarg);
      continue;
    }
    print_usage();
    return 1;
  }

  if (fname == NULL) {
    print_usage();
  }

  cudaSetDevice(gpu_num);

  // TODO: Add more datatype options if needed
  if (dtype == "int") {
    run_benchmark<int32_t>(
        fname,
        size,
        verbose_memory);
  } else if (dtype == "long") {
    run_benchmark<int64_t>(
        fname,
        size,
        verbose_memory);
  } else if (dtype == "short") {
    run_benchmark<int16_t>(
        fname,
        size,
        verbose_memory);
  } else if (dtype == "int8") {
    run_benchmark<int8_t>(
        fname,
        size,
        verbose_memory);
  } else {
    print_usage();
  }

  return 0;
}
