/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp/snappy.h"
#include "benchmark_common.h"

#include "cuda_runtime.h"

#include <string>
#include <vector>
#include <numeric>
#include <fstream>
#include <cstring>

using namespace nvcomp;

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      exit(0);                                                                 \
    }                                                                          \
  } while (0)

namespace
{

constexpr const size_t CHUNK_SIZE = 1 << 16;
constexpr const int DEFAULT_WARMUP_COUNT = 10;
constexpr const int DEFAULT_ITERATIONS_COUNT = 10;
constexpr const int DEFAULT_PAGE_SIZE = 65536;

void print_usage()
{
  printf("Usage: benchmark_snappy_chunked [OPTIONS]\n");
  printf("  %-35s GPU device number (default 0)\n", "-g, --gpu");
  printf("  %-35s Input file path, you can specify multiple files this way\n", "-f, --input_file");
  printf("  %-35s Warm up benchmark (default %d)\n", "-w, --warmup_count", DEFAULT_WARMUP_COUNT);
  printf("  %-35s Average multiple kernel runtimes (default %d)\n", "-i, --iterations_count", DEFAULT_ITERATIONS_COUNT);
  printf("  %-35s Clone uncompressed chunks multiple times (default 0)\n", "-x, --duplicate_data");
  printf(
      "  %-35s Output in column/csv format (default comma)\n", "--csv-output");
  printf("  %-35s Use tab separator for the output (default comma)\n", "-t, --tab");
  printf("  %-35s Files(s) contain pages, each prefixed with int64 size\n", "-s, --file_with_page_sizes");
  printf("  %-35s Page size to use when splitting uncompressed data (default %d)\n", "-p, --page_size", DEFAULT_PAGE_SIZE);
  exit(1);
}

void run_benchmark(
    const std::vector<std::vector<uint8_t>>& uncompressed_data,
    int warmup_count,
    int iterations_count,
    char separator,
    const bool csv_output)
{
  size_t batch_size = uncompressed_data.size();

  // prepare input and output on host
  size_t total_bytes_uncompressed = 0;
  std::vector<size_t> batch_bytes_host(batch_size);
  size_t max_batch_bytes_uncompressed = 0; 
  for (size_t i = 0; i < batch_size; ++i) {
    batch_bytes_host[i] = uncompressed_data[i].size();
    total_bytes_uncompressed += uncompressed_data[i].size();
    if (batch_bytes_host[i] > max_batch_bytes_uncompressed)
      max_batch_bytes_uncompressed = batch_bytes_host[i];
  }

  if (csv_output) {
    std::cout << separator << total_bytes_uncompressed;
  }

  size_t * batch_bytes_device;
  CUDA_CHECK(cudaMalloc((void **)(&batch_bytes_device), sizeof(size_t) * batch_bytes_host.size()));
  CUDA_CHECK(cudaMemcpy(batch_bytes_device, batch_bytes_host.data(), sizeof(size_t) * batch_bytes_host.size(), cudaMemcpyHostToDevice));

  std::vector<const uint8_t *> input_host(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    input_host[i] = uncompressed_data[i].data();
  }

  std::vector<uint8_t *> output_host(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    output_host[i] = (uint8_t *)malloc(batch_bytes_host[i]);
  }

  // prepare gpu buffers
  std::vector<void *> d_in_data(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&d_in_data[i], batch_bytes_host[i]));
    CUDA_CHECK(cudaMemcpy(
        d_in_data[i],
        input_host[i],
        batch_bytes_host[i],
        cudaMemcpyHostToDevice));
  }
  void** d_in_data_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_in_data_device), sizeof(void *) * d_in_data.size()));
  CUDA_CHECK(cudaMemcpy(d_in_data_device, d_in_data.data(), sizeof(void *) * d_in_data.size(), cudaMemcpyHostToDevice));


  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompStatus_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedSnappyCompressGetTempSize(
      batch_size,
      max_batch_bytes_uncompressed,
      &comp_temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t comp_out_bytes;
  status = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
      max_batch_bytes_uncompressed,
      &comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  std::vector<void *> d_comp_out(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&d_comp_out[i], comp_out_bytes));
  }

  void** d_comp_out_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_comp_out_device), sizeof(void *) * d_comp_out.size()));
  CUDA_CHECK(cudaMemcpy(d_comp_out_device, d_comp_out.data(), sizeof(void *) * d_comp_out.size(), cudaMemcpyHostToDevice));

  size_t * comp_out_bytes_device;
  CUDA_CHECK(cudaMalloc((void **)(&comp_out_bytes_device), sizeof(size_t) * batch_size));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // warmup
  for(int i = 0; i < warmup_count; ++i) {
    status = nvcompBatchedSnappyCompressAsync(
        (const void* const*)d_in_data_device,
        batch_bytes_device,
        max_batch_bytes_uncompressed,
        batch_size,
        d_comp_temp,
        comp_temp_bytes,
        d_comp_out_device,
        comp_out_bytes_device,
        nvcompBatchedSnappyDefaultOpts,
        stream);
    REQUIRE(status == nvcompSuccess);
  }
  CUDA_CHECK(cudaEventRecord(start, stream));
  for(int i = 0; i < iterations_count; ++i) {
    status = nvcompBatchedSnappyCompressAsync(
        (const void* const*)d_in_data_device,
        batch_bytes_device,
        max_batch_bytes_uncompressed,
        batch_size,
        d_comp_temp,
        comp_temp_bytes,
        d_comp_out_device,
        comp_out_bytes_device,
        nvcompBatchedSnappyDefaultOpts,
        stream);
    REQUIRE(status == nvcompSuccess);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float elapsedTime;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  std::vector<size_t> comp_out_bytes_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
    comp_out_bytes_host.data(),
    comp_out_bytes_device,
    sizeof(size_t) * batch_size,
    cudaMemcpyDeviceToHost));
  size_t total_bytes_compressed = std::accumulate(comp_out_bytes_host.begin(), comp_out_bytes_host.end(), (size_t)0);
  if (csv_output) {
    std::cout << separator << total_bytes_compressed;
    std::cout << separator << std::fixed << std::setprecision(2)
              << (double)total_bytes_uncompressed / total_bytes_compressed;
    std::cout << separator
              << (total_bytes_compressed + total_bytes_uncompressed)
                     / (elapsedTime * 0.001F / iterations_count) / 1.0e+9F;
  } else {
    std::cout << "uncompressed (B): " << total_bytes_uncompressed << std::endl;
    std::cout << "comp_size: " << total_bytes_compressed
              << ", compressed ratio: " << std::fixed << std::setprecision(2)
              << (double)total_bytes_uncompressed / total_bytes_compressed
              << std::endl;
    std::cout << "compression throughput (GB/s): "
              << (double)total_bytes_uncompressed
                     / (1.0e6 * (elapsedTime / iterations_count))
              << std::endl;
  }

  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaFree(d_in_data_device));
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaFree(d_in_data[i]));
  }

  size_t temp_bytes;
  status = nvcompBatchedSnappyDecompressGetTempSize(
      batch_size,
      max_batch_bytes_uncompressed,
      &temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

  std::vector<void *> d_decomp_out(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    CUDA_CHECK(cudaMalloc(&d_decomp_out[i], max_batch_bytes_uncompressed));
  }
  void** d_decomp_out_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_decomp_out_device), sizeof(void *) * d_decomp_out.size()));
  CUDA_CHECK(cudaMemcpy(d_decomp_out_device, d_decomp_out.data(), sizeof(void *) * d_decomp_out.size(), cudaMemcpyHostToDevice));

  nvcompStatus_t * device_statuses;
  CUDA_CHECK(cudaMalloc(
      (void**)(&device_statuses), batch_size * sizeof(nvcompStatus_t)));

  for(int i = 0; i < warmup_count; ++i) {
    status = nvcompBatchedSnappyDecompressAsync(
        (const void* const*)d_comp_out_device,
        comp_out_bytes_device,
        batch_bytes_device,
        batch_bytes_device,
        batch_size,
        temp_ptr,
        temp_bytes,
        (void* const*)d_decomp_out_device,
        device_statuses,
        stream);
    REQUIRE(status == nvcompSuccess);
  }
  CUDA_CHECK(cudaEventRecord(start, stream));
  for(int i = 0; i < iterations_count; ++i) {
    status = nvcompBatchedSnappyDecompressAsync(
        (const void* const*)d_comp_out_device,
        comp_out_bytes_device,
        batch_bytes_device,
        batch_bytes_device,
        batch_size,
        temp_ptr,
        temp_bytes,
        (void* const*)d_decomp_out_device,
        device_statuses,
        stream);
    REQUIRE(status == nvcompSuccess);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  if (csv_output) {
    std::cout << separator
              << (total_bytes_compressed + total_bytes_uncompressed)
                     / (elapsedTime * 0.001F / iterations_count) / 1.0e+9F;
  } else {
    std::cout << "decompression throughput (GB/s): "
              << (double)total_bytes_uncompressed
                     / (1.0e6 * (elapsedTime / iterations_count))
              << std::endl;
  }

  CUDA_CHECK(cudaFree(temp_ptr));
  CUDA_CHECK(cudaFree(d_comp_out_device));
  CUDA_CHECK(cudaFree(comp_out_bytes_device));
  CUDA_CHECK(cudaFree(batch_bytes_device));
  CUDA_CHECK(cudaFree(device_statuses));

  for (size_t i = 0; i < batch_size; i++) {
    CUDA_CHECK(cudaMemcpy(output_host[i], d_decomp_out[i], batch_bytes_host[i], cudaMemcpyDeviceToHost));
    // Verify correctness
    for (size_t j = 0; j < batch_bytes_host[i]; ++j) {
      if (output_host[i][j] != input_host[i][j])
        std::cout << "Mismatch at batch # " << i << ", element " << j
          << ", reference value = " << (unsigned int)input_host[i][j]
          << ", actual value = " << (unsigned int)output_host[i][j] << std::endl;
      REQUIRE(output_host[i][j] == input_host[i][j]);
    }
  }

  for (size_t i = 0; i < batch_size; i++) {
    CUDA_CHECK(cudaFree(d_comp_out[i]));
    CUDA_CHECK(cudaFree(d_decomp_out[i]));
    free(output_host[i]);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

std::vector<std::vector<uint8_t>> readFileWithPageSizes(const std::string& filename)
{
  std::vector<std::vector<uint8_t>> res;

  std::ifstream fin(filename, std::ifstream::binary);

  while (!fin.eof()) {
    uint64_t chunk_size;
    fin.read((char *)(&chunk_size), sizeof(uint64_t));
    if (fin.eof())
      break;
    res.emplace_back(chunk_size);
    fin.read((char *)(res.back().data()), chunk_size);
  }

  return res;
}

std::vector<std::vector<uint8_t>> readFile(const std::string& filename, int page_size)
{
  std::vector<std::vector<uint8_t>> res;

  std::ifstream fin(filename, std::ifstream::binary);

  while (!fin.eof()) {
    res.emplace_back(page_size);
    fin.read((char *)(res.back().data()), page_size);
    if (fin.gcount() < page_size)
      res.back().resize(fin.gcount());
  }

  return res;
}

} // namespace

int main(int argc, char* argv[])
{
  int gpu_num = 0;
  std::vector<std::string> input_files;
  int warmup_count = DEFAULT_WARMUP_COUNT;
  int iterations_count = DEFAULT_ITERATIONS_COUNT;
  int duplicate_data = 0;
  char separator = ',';
  int page_size = DEFAULT_PAGE_SIZE;
  bool file_with_page_sizes = false;
  bool csv_output = false;

  char** argv_end = argv + argc;
  argv += 1;
  while (argv != argv_end) {
    char* arg = *argv++;
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-?") == 0) {
      print_usage();
      return 1;
    }
    if (strcmp(arg, "--tab") == 0 || strcmp(arg, "-t") == 0) {
      separator = '\t';
      continue;
    }
    if (strcmp(arg, "--file_with_page_sizes") == 0 || strcmp(arg, "-s") == 0) {
      file_with_page_sizes = true;
      continue;
    }
    if (strcmp(arg, "--csv-output") == 0) {
      csv_output = true;
      continue;
    }

    // all arguments below require at least a second value in argv
    if (argv >= argv_end) {
      print_usage();
      return 1;
    }

    char* optarg = *argv++;
    if (strcmp(arg, "--gpu") == 0 || strcmp(arg, "-g") == 0) {
      gpu_num = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--input_file") == 0 || strcmp(arg, "-f") == 0) {
      input_files.push_back(optarg);
      continue;
    }
    if (strcmp(arg, "--warmup_count") == 0 || strcmp(arg, "-w") == 0) {
      warmup_count = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--iterations_count") == 0 || strcmp(arg, "-i") == 0) {
      iterations_count = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--duplicate_data") == 0 || strcmp(arg, "-x") == 0) {
      duplicate_data = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--page_size") == 0 || strcmp(arg, "-p") == 0) {
      page_size = atoi(optarg);
      continue;
    }
    print_usage();
    return 1;
  }

  if (input_files.empty()) {
    printf("No input files specified\n");
    print_usage();
    exit(1);
  }

  cudaSetDevice(gpu_num);

  if (csv_output) {
    std::cout << "File" << separator << "Duplicate data" << separator
              << "Size in MB" << separator << "Pages" << separator
              << "Avg page size in KB" << separator << "Max page size in KB"
              << separator << "Ucompressed size in bytes" << separator
              << "Compressed size in bytes" << separator << "Compression ratio"
              << separator << "Compression throughput read+write in GB/s"
              << separator << "Decompression throughput read+write in GB/s"
              << std::endl;
  }

  for(const std::string& input_file: input_files) {
    if (csv_output) {
      std::cout << input_file;
    } else {
      std::cout << "----------" << std::endl;
      std::cout << "file: " << input_file << std::endl;
    }

    std::vector<std::vector<uint8_t>> uncompressed_data = file_with_page_sizes ? 
      readFileWithPageSizes(input_file) : readFile(input_file, page_size);

    size_t original_chunks = uncompressed_data.size();
    {
      if (csv_output) {
        std::cout << separator << duplicate_data;
      }

      uncompressed_data.resize(original_chunks * (duplicate_data + 1));
      for(int i = 0; i < duplicate_data; ++i)
        for(size_t j = 0; j < original_chunks; ++j)
          uncompressed_data[(i + 1) * original_chunks + j] = uncompressed_data[j];
    }

    uint64_t total_size = std::accumulate(uncompressed_data.begin(), uncompressed_data.end(), (uint64_t)0,
      [] (uint64_t accum, const std::vector<uint8_t>& chunk) { return accum + chunk.size(); } );
    uint64_t max_size = std::accumulate(uncompressed_data.begin(), uncompressed_data.end(), (uint64_t)0,
      [] (uint64_t accum, const std::vector<uint8_t>& chunk) { return std::max<uint64_t>(accum, chunk.size()); } );

    if (csv_output) {
      std::cout << separator << std::fixed << std::setprecision(2)
                << (total_size / 1.0e+6F);
      std::cout << separator << uncompressed_data.size();
      std::cout << separator << total_size / uncompressed_data.size() / 1.0e+3F;
      std::cout << separator << max_size / 1.0e+3F;
    }

    run_benchmark(
        uncompressed_data,
        warmup_count,
        iterations_count,
        separator,
        csv_output);

    if (csv_output) {
      std::cout << std::endl;
    }
  }

  return 0;
}
