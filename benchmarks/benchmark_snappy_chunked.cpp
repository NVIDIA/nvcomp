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

#include <getopt.h>
#include <string>
#include <vector>
#include <numeric>
#include <fstream>

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
constexpr const int DEFAULT_MAX_BYTE_VALUE = 255;

void print_usage()
{
  printf("Usage: benchmark_binary [OPTIONS]\n");
  printf("  %-35s GPU device number (default 0)\n", "-g, --gpu");
  printf("  %-35s Input file path\n", "-f, --input_file");
  printf("  %-35s Warm up benchmark (default %d)\n", "-w, --warmup_count", DEFAULT_WARMUP_COUNT);
  printf("  %-35s Average multiple kernel runtimes (default %d)\n", "-i, --iterations_count", DEFAULT_ITERATIONS_COUNT);
  printf("  %-35s Maximum value for the bytes of uncompressed data (default %d)\n", "-m, --max_byte", DEFAULT_MAX_BYTE_VALUE);
  printf("  %-35s Clone uncompressed chunks multiple times (default 0)\n", "-x, --duplicate_data");
  exit(1);
}

void run_benchmark(const std::vector<std::vector<uint8_t>>& uncompressed_data, int warmup_count, int iterations_count)
{
  size_t batch_size = uncompressed_data.size();

  // prepare input and output on host
  size_t total_bytes_uncompressed = 0;
  size_t batch_bytes_host[batch_size];
  size_t max_batch_bytes_uncompressed = 0; 
  for (size_t i = 0; i < batch_size; ++i) {
    batch_bytes_host[i] = uncompressed_data[i].size();
    total_bytes_uncompressed += uncompressed_data[i].size();
    if (batch_bytes_host[i] > max_batch_bytes_uncompressed)
      max_batch_bytes_uncompressed = batch_bytes_host[i];
  }

  std::cout << "uncompressed (B): " << total_bytes_uncompressed << std::endl;
  std::cout << "chunks " << batch_size << std::endl;

  size_t * batch_bytes_device;
  CUDA_CHECK(cudaMalloc((void **)(&batch_bytes_device), sizeof(batch_bytes_host)));
  cudaMemcpy(batch_bytes_device, batch_bytes_host, sizeof(batch_bytes_host), cudaMemcpyHostToDevice);

  const uint8_t * input_host[batch_size];
  for (size_t i = 0; i < batch_size; ++i) {
    input_host[i] = uncompressed_data[i].data();
  }

  uint8_t * output_host[batch_size];
  for (size_t i = 0; i < batch_size; ++i) {
    output_host[i] = (uint8_t *)malloc(batch_bytes_host[i]);
  }

  // prepare gpu buffers
  void* d_in_data[batch_size];
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&d_in_data[i], batch_bytes_host[i]));
    CUDA_CHECK(cudaMemcpy(
        d_in_data[i],
        input_host[i],
        batch_bytes_host[i],
        cudaMemcpyHostToDevice));
  }
  void** d_in_data_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_in_data_device), sizeof(d_in_data)));
  cudaMemcpy(d_in_data_device, d_in_data, sizeof(d_in_data), cudaMemcpyHostToDevice);


  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompError_t status;

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
  status = nvcompBatchedSnappyCompressGetOutputSize(
      max_batch_bytes_uncompressed,
      &comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_out[batch_size];
  for (size_t i = 0; i < batch_size; ++i) {
    CUDA_CHECK(cudaMalloc(&d_comp_out[i], comp_out_bytes));
  }

  void** d_comp_out_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_comp_out_device), sizeof(d_comp_out)));
  cudaMemcpy(d_comp_out_device, d_comp_out, sizeof(d_comp_out), cudaMemcpyHostToDevice);

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
        batch_size,
        d_comp_temp,
        comp_temp_bytes,
        d_comp_out_device,
        comp_out_bytes_device,
        stream);
    REQUIRE(status == nvcompSuccess);
  }
  CUDA_CHECK(cudaEventRecord(start, 0));
  for(int i = 0; i < iterations_count; ++i) {
    status = nvcompBatchedSnappyCompressAsync(
        (const void* const*)d_in_data_device,
        batch_bytes_device,
        batch_size,
        d_comp_temp,
        comp_temp_bytes,
        d_comp_out_device,
        comp_out_bytes_device,
        stream);
    REQUIRE(status == nvcompSuccess);
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  std::vector<size_t> comp_out_bytes_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
    comp_out_bytes_host.data(),
    comp_out_bytes_device,
    sizeof(size_t) * batch_size,
    cudaMemcpyDeviceToHost));
  size_t total_bytes_compressed = std::accumulate(comp_out_bytes_host.begin(), comp_out_bytes_host.end(), (size_t)0);
  std::cout << "comp_size: " << total_bytes_compressed
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes_uncompressed / total_bytes_compressed << std::endl;
  std::cout << "compression throughput read+write (GB/s): " << (total_bytes_compressed + total_bytes_uncompressed) / (elapsedTime * 0.001F / iterations_count) / 1.0e+9F
            << std::endl;

  cudaFree(d_comp_temp);
  cudaFree(d_in_data_device);
  for (size_t i = 0; i < batch_size; ++i) {
    cudaFree(d_in_data[i]);
  }

  size_t temp_bytes;
  status = nvcompBatchedSnappyDecompressGetTempSize(
      batch_size,
      max_batch_bytes_uncompressed,
      &temp_bytes);
  REQUIRE(status == nvcompSuccess);

  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));

  void* d_decomp_out[batch_size];
  for (int i = 0; i < batch_size; i++) {
    CUDA_CHECK(cudaMalloc(&d_decomp_out[i], max_batch_bytes_uncompressed));
  }
  void** d_decomp_out_device;
  CUDA_CHECK(cudaMalloc((void **)(&d_decomp_out_device), sizeof(d_decomp_out)));
  cudaMemcpy(d_decomp_out_device, d_decomp_out, sizeof(d_decomp_out), cudaMemcpyHostToDevice);

  for(int i = 0; i < warmup_count; ++i) {
    status = nvcompBatchedSnappyDecompressAsync(
        (const void* const*)d_comp_out_device,
        comp_out_bytes_device,
        batch_bytes_device,
        batch_size,
        temp_ptr,
        temp_bytes,
        (void* const*)d_decomp_out_device,
        stream);
    REQUIRE(status == nvcompSuccess);
  }
  CUDA_CHECK(cudaEventRecord(start, 0));
  for(int i = 0; i < iterations_count; ++i) {
    status = nvcompBatchedSnappyDecompressAsync(
        (const void* const*)d_comp_out_device,
        comp_out_bytes_device,
        batch_bytes_device,
        batch_size,
        temp_ptr,
        temp_bytes,
        (void* const*)d_decomp_out_device,
        stream);
    REQUIRE(status == nvcompSuccess);
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  elapsedTime;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  std::cout << "decompression throughput read+write (GB/s): "
            << (total_bytes_compressed + total_bytes_uncompressed) / (elapsedTime * 0.001F / iterations_count) / 1.0e+9F << std::endl;

  cudaFree(temp_ptr);
  cudaFree(d_comp_out_device);
  cudaFree(comp_out_bytes_device);
  cudaFree(batch_bytes_device);

  for (int i = 0; i < batch_size; i++) {
    cudaMemcpy(
        output_host[i],
        d_decomp_out[i],
        batch_bytes_host[i],
        cudaMemcpyDeviceToHost);
    // Verify correctness
    for (size_t j = 0; j < batch_bytes_host[i]; ++j) {
      if (output_host[i][j] != input_host[i][j])
        std::cout << "Mismatch at batch # " << i << ", element " << j
          << ", reference value = " << (unsigned int)input_host[i][j]
          << ", actual value = " << (unsigned int)output_host[i][j] << std::endl;
      REQUIRE(output_host[i][j] == input_host[i][j]);
    }
  }

  for (int i = 0; i < batch_size; i++) {
    cudaFree(d_comp_out[i]);
    cudaFree(d_decomp_out[i]);
    free(output_host[i]);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

std::vector<std::vector<uint8_t>> readFile(const std::string& filename)
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

} // namespace

int main(int argc, char* argv[])
{
  int gpu_num = 0;
  std::string input_file;
  int warmup_count = DEFAULT_WARMUP_COUNT;
  int iterations_count = DEFAULT_ITERATIONS_COUNT;
  int duplicate_data = 0;

  // Parse command-line arguments
  while (1) {
    int option_index = 0;
    static struct option long_options[]{{"gpu", required_argument, 0, 'g'},
                                        {"input_file", required_argument, 0, 'f'},
                                        {"warmup_count", required_argument, 0, 'w'},
                                        {"iterations_count", required_argument, 0, 'i'},
                                        {"duplicate_data", required_argument, 0, 'x'},
                                        {"help", no_argument, 0, '?'}};
    int c;
    opterr = 0;
    c = getopt_long(argc, argv, "g:f:w:i:m:x:", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 'g':
      gpu_num = atoi(optarg);
      break;
    case 'f':
      input_file = optarg;
      break;
    case 'w':
      warmup_count = atoi(optarg);
      break;
    case 'i':
      iterations_count = atoi(optarg);
      break;
    case 'x':
      duplicate_data = atoi(optarg);
      break;
    case '?':
    default:
      print_usage();
      exit(1);
    }
  }

  std::cout << "----------" << std::endl;
  std::cout << "Input file = " << input_file << std::endl;

  std::vector<std::vector<uint8_t>> uncompressed_data = readFile(input_file);

  size_t original_chunks = uncompressed_data.size();
  if (duplicate_data > 0) {
    std::cout << "Chunks duplicated " << duplicate_data << "x" << std::endl;

    uncompressed_data.resize(original_chunks * (duplicate_data + 1));
    for(int i = 0; i < duplicate_data; ++i)
      for(int j = 0; j < original_chunks; ++j)
        uncompressed_data[(i + 1) * original_chunks + j] = uncompressed_data[j];
  }

  cudaSetDevice(gpu_num);

  uint64_t total_size = std::accumulate(uncompressed_data.begin(), uncompressed_data.end(), (uint64_t)0,
    [] (uint64_t accum, const std::vector<uint8_t>& chunk) { return accum + chunk.size(); } );
  uint64_t max_size = std::accumulate(uncompressed_data.begin(), uncompressed_data.end(), (uint64_t)0,
    [] (uint64_t accum, const std::vector<uint8_t>& chunk) { return std::max<uint64_t>(accum, chunk.size()); } );
  std::cout << "Size " << std::fixed << std::setprecision(2) << (total_size / 1.0e+6F) << "MB, in "
    << uncompressed_data.size() << " chunks, " << total_size / uncompressed_data.size() / 1.0e+3F << " KB per chunk, max chunk size "
    << max_size / 1.0e+3F << " KB" << std::endl;

  run_benchmark(uncompressed_data, warmup_count, iterations_count);

  std::cout << "----------" << std::endl;

  return 0;
}
