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

#include "nvcomp/lz4.hpp"

#include "benchmark_common.h"

#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

using namespace nvcomp;

static size_t compute_batch_size(
    const std::vector<std::vector<char>>& data, const size_t chunk_size)
{
  size_t batch_size = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    batch_size += num_chunks;
  }

  return batch_size;
}

std::vector<size_t> compute_chunk_sizes(
    const std::vector<std::vector<char>>& data,
    const size_t batch_size,
    const size_t chunk_size)
{
  std::vector<size_t> sizes(batch_size, chunk_size);

  size_t offset = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    if (data[i].size() % chunk_size != 0) {
      sizes[offset] = data[i].size() % chunk_size;
    }
    offset += num_chunks;
  }
  return sizes;
}

class BatchData
{
public:
  BatchData(
      const std::vector<std::vector<char>>& host_data,
      const size_t chunk_size) :
      m_ptrs(), m_sizes(), m_data(), m_size(0)
  {
    m_size = compute_batch_size(host_data, chunk_size);

    m_data = thrust::device_vector<uint8_t>(chunk_size * size());

    std::vector<void*> uncompressed_ptrs(size());
    for (size_t i = 0; i < size(); ++i) {
      uncompressed_ptrs[i] = static_cast<void*>(data() + chunk_size * i);
    }

    m_ptrs = thrust::device_vector<void*>(uncompressed_ptrs);
    std::vector<size_t> sizes
        = compute_chunk_sizes(host_data, size(), chunk_size);
    m_sizes = thrust::device_vector<size_t>(sizes);

    // copy data to GPU
    size_t offset = 0;
    for (size_t i = 0; i < host_data.size(); ++i) {
      CUDA_CHECK(cudaMemcpy(
          uncompressed_ptrs[offset],
          host_data[i].data(),
          host_data[i].size(),
          cudaMemcpyHostToDevice));

      const size_t num_chunks
          = (host_data[i].size() + chunk_size - 1) / chunk_size;
      offset += num_chunks;
    }
  }

  BatchData(const size_t max_output_size, const size_t batch_size) :
      m_ptrs(), m_sizes(), m_data(), m_size(batch_size)
  {
    m_data = thrust::device_vector<uint8_t>(max_output_size * size());

    std::vector<size_t> sizes(size(), max_output_size);
    m_sizes = thrust::device_vector<size_t>(sizes);

    std::vector<void*> ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      ptrs[i] = data() + max_output_size * i;
    }
    m_ptrs = thrust::device_vector<void*>(ptrs);
  }

  BatchData(BatchData&& other) = default;

  // disable copying
  BatchData(const BatchData& other) = delete;
  BatchData& operator=(const BatchData& other) = delete;

  void** ptrs()
  {
    return m_ptrs.data().get();
  }

  size_t* sizes()
  {
    return m_sizes.data().get();
  }

  uint8_t* data()
  {
    return m_data.data().get();
  }

  size_t size() const
  {
    return m_size;
  }

private:
  thrust::device_vector<void*> m_ptrs;
  thrust::device_vector<size_t> m_sizes;
  thrust::device_vector<uint8_t> m_data;
  size_t m_size;
};

// Benchmark performance from the binary data file fname
static void
run_benchmark(const std::vector<std::vector<char>>& data, const bool warmup)
{
  size_t total_bytes = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
  }

  if (!warmup) {
    std::cout << "----------" << std::endl;
    std::cout << "files: " << data.size() << std::endl;
    std::cout << "uncompressed (B): " << total_bytes << std::endl;
  }

  const size_t chunk_size = 1 << 16;

  // build up metadata
  BatchData input_data(data, chunk_size);

  // compression
  nvcompStatus_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedLZ4CompressGetTempSize(
      input_data.size(), chunk_size, &comp_temp_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedLZ4CompressGetTempSize() failed.");
  }

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_out_bytes;
  status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      chunk_size, &max_out_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedLZ4GetMaxOutputChunkSize() failed.");
  }

  BatchData compress_data(max_out_bytes, input_data.size());

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);

  nvcomp_lz4_lowlevel_opt_type fmt_opts;

  status = nvcompBatchedLZ4CompressAsync(
      input_data.ptrs(),
      input_data.sizes(),
      chunk_size,
      input_data.size(),
      d_comp_temp,
      comp_temp_bytes,
      compress_data.ptrs(),
      compress_data.sizes(),
      &fmt_opts,
      stream);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedLZ4CompressAsync() failed.");
  }

  cudaEventRecord(end, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // free compression memory
  cudaFree(d_comp_temp);

  float ms;
  cudaEventElapsedTime(&ms, start, end);

  if (!warmup) {
    // compute compression ratio
    std::vector<size_t> compressed_sizes_host(compress_data.size());
    cudaMemcpy(
        compressed_sizes_host.data(),
        compress_data.sizes(),
        compress_data.size() * sizeof(*compress_data.sizes()),
        cudaMemcpyDeviceToHost);

    size_t comp_bytes = 0;
    for (const size_t s : compressed_sizes_host) {
      comp_bytes += s;
    }

    std::cout << "comp_size: " << comp_bytes
              << ", compressed ratio: " << std::fixed << std::setprecision(2)
              << (double)total_bytes / comp_bytes << std::endl;
    std::cout << "compression throughput (GB/s): "
              << (double)total_bytes / (1.0e6 * ms) << std::endl;
  }

  // overwrite our uncompressed data so we can test for correctness
  CUDA_CHECK(cudaMemset(input_data.data(), 0, chunk_size * input_data.size()));

  // LZ4 decompression
  size_t decomp_temp_bytes;
  status = nvcompBatchedLZ4DecompressGetTempSize(
      compress_data.size(), chunk_size, &decomp_temp_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedLZ4DecompressGetTempSize() failed.");
  }

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  size_t* d_decomp_sizes;
  CUDA_CHECK(
      cudaMalloc((void**)&d_decomp_sizes, input_data.size() * sizeof(size_t)));

  nvcompStatus_t* d_status_ptrs;
  CUDA_CHECK(cudaMalloc(
      (void**)&d_status_ptrs, input_data.size() * sizeof(nvcompStatus_t)));

  cudaEventRecord(start, stream);

  status = nvcompBatchedLZ4DecompressAsync(
      compress_data.ptrs(),
      compress_data.sizes(),
      input_data.sizes(),
      d_decomp_sizes,
      compress_data.size(),
      d_decomp_temp,
      decomp_temp_bytes,
      input_data.ptrs(),
      d_status_ptrs,
      stream);
  benchmark_assert(
      status == nvcompSuccess,
      "nvcompBatchedLZ4DecompressAsync() not successful");

  cudaEventRecord(end, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEventElapsedTime(&ms, start, end);

  if (!warmup) {
    std::cout << "decompression throughput (GB/s): "
              << (double)total_bytes / (1.0e6 * ms) << std::endl;
  }

  cudaFree(d_decomp_temp);
  cudaFree(d_decomp_sizes);
  cudaFree(d_status_ptrs);

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
  std::vector<std::string> file_names(argc - 1);

  if (argc == 1) {
    std::cerr << "Must specify at least one file." << std::endl;
    return 1;
  }

  // if `-f` is speficieid, assume single file mode
  if (strcmp(argv[1], "-f") == 0) {
    if (argc == 2) {
      std::cerr << "Missing file name following '-f'" << std::endl;
      return 1;
    } else if (argc > 3) {
      std::cerr << "Unknown extra arguments with '-f'." << std::endl;
      return 1;
    }

    file_names = {argv[2]};
  } else {
    // multi-file mode
    for (int i = 1; i < argc; ++i) {
      file_names[i - 1] = argv[i];
    }
  }

  auto data = multi_file(file_names);

  // one warmup to allow cuda to initialize
  run_benchmark(data, true);

  // second run to report times
  run_benchmark(data, false);

  return 0;
}
