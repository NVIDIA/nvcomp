/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "gdeflate.h"
#include "nvcomp/gdeflate.h"

#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string.h>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

#ifdef ENABLE_GDEFLATE
#define CHECK_NVCOMP_STATUS(status)                                            \
  if ((status) != nvcompSuccess)                                               \
    throw std::runtime_error("Failed to decompress data");
#else
#define CHECK_NVCOMP_STATUS(status)                                            \
  if ((status) != nvcompSuccess)                                               \
    throw std::runtime_error("nvcomp not configured with gdeflate support");
#endif

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw;                                                                   \
    }                                                                          \
  } while (0);

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
    offset += num_chunks;
    if (data[i].size() % chunk_size != 0) {
      sizes[offset-1] = data[i].size() % chunk_size;
    }
  }
  return sizes;
}

std::vector<void*> get_input_ptrs(
    const std::vector<std::vector<char>>& data,
    const size_t batch_size,
    const size_t chunk_size)
{
  std::vector<void*> input_ptrs(batch_size);
  size_t chunk = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    for (size_t j = 0; j < num_chunks; ++j)
      input_ptrs[chunk++] = const_cast<void*>(static_cast<const void*>(data[i].data() + j*chunk_size));
  }
  return input_ptrs;
}

class BatchDataCPU
{
public:
  BatchDataCPU(
      const std::vector<std::vector<char>>& host_data,
      const size_t chunk_size) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(0)
  {
    m_size = compute_batch_size(host_data, chunk_size);
    m_sizes = compute_chunk_sizes(host_data, m_size, chunk_size);

    size_t data_size = std::accumulate(m_sizes.begin(), m_sizes.end(), static_cast<size_t>(0));
    m_data = std::vector<uint8_t>(data_size);

    size_t offset = 0;
    m_ptrs = std::vector<void*>(size());
    for (size_t i = 0; i < size(); ++i) {
      m_ptrs[i] = data() + offset;
      offset += m_sizes[i];
    }

    std::vector<void*> src = get_input_ptrs(host_data, size(), chunk_size);
    for (size_t i = 0; i < size(); ++i)
      std::memcpy(m_ptrs[i], src[i], m_sizes[i]);
  }

  BatchDataCPU(const size_t max_output_size, const size_t batch_size) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(batch_size)
  {
    m_data = std::vector<uint8_t>(max_output_size * size());

    m_sizes = std::vector<size_t>(size(), max_output_size);

    m_ptrs = std::vector<void*>(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      m_ptrs[i] = data() + max_output_size * i;
    }
  }

  BatchDataCPU(BatchDataCPU&& other) = default;

  // disable copying
  BatchDataCPU(const BatchDataCPU& other) = delete;
  BatchDataCPU& operator=(const BatchDataCPU& other) = delete;

  uint8_t* data() { return m_data.data(); }
  const uint8_t* data() const { return m_data.data(); }

  void** ptrs() { return m_ptrs.data(); }
  const void* const * ptrs() const { return m_ptrs.data(); }

  size_t* sizes() { return m_sizes.data(); }
  const size_t* sizes() const { return m_sizes.data(); }

  size_t size() const { return m_size; }

private:
  std::vector<void*> m_ptrs;
  std::vector<size_t> m_sizes;
  std::vector<uint8_t> m_data;
  size_t m_size;
};

class BatchData
{
public:
  BatchData(
      const std::vector<std::vector<char>>& host_data,
      const size_t chunk_size) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(0)
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

  BatchData(const BatchDataCPU& batch_data, bool copy_data=false) :
    m_ptrs(),
    m_sizes(),
    m_data(),
    m_size()
  {
    m_size = batch_data.size();
    m_sizes = thrust::device_vector<size_t>(batch_data.sizes(), batch_data.sizes() + size());

    size_t data_size = std::accumulate(batch_data.sizes(), batch_data.sizes() + size(), static_cast<size_t>(0));
    m_data = thrust::device_vector<uint8_t>(data_size);

    size_t offset = 0;
    std::vector<void*> ptrs(size());
    for (size_t i = 0; i < size(); ++i) {
      ptrs[i] = data() + offset;
      offset += batch_data.sizes()[i];
    }
    m_ptrs = thrust::device_vector<void*>(ptrs);

    if (copy_data) {
      const void* const * src = batch_data.ptrs();
      const size_t* bytes = batch_data.sizes();
      for (size_t i = 0; i < size(); ++i)
        CUDA_CHECK(cudaMemcpy(ptrs[i], src[i], bytes[i], cudaMemcpyHostToDevice));
    }
  }

  BatchData(const size_t max_output_size, const size_t batch_size) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(batch_size)
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

  uint8_t* data() { return m_data.data().get(); }
  const uint8_t* data() const { return m_data.data().get(); }

  void** ptrs() { return m_ptrs.data().get(); }
  const void* const * ptrs() const { return m_ptrs.data().get(); }

  size_t* sizes() { return m_sizes.data().get(); }
  const size_t* sizes() const { return m_sizes.data().get(); }

  size_t size() const { return m_size; }

private:
  thrust::device_vector<void*> m_ptrs;
  thrust::device_vector<size_t> m_sizes;
  thrust::device_vector<uint8_t> m_data;
  size_t m_size;
};

inline bool operator==(
    const BatchDataCPU& lhs,
    const BatchData& rhs)
{
  size_t batch_size = lhs.size();

  if (lhs.size() != rhs.size()) return false;

  std::vector<size_t> rhs_sizes(rhs.size());
  CUDA_CHECK(cudaMemcpy(rhs_sizes.data(), rhs.sizes(), rhs.size()*sizeof(size_t), cudaMemcpyDeviceToHost));

  std::vector<void*> rhs_ptrs(rhs.size());
  CUDA_CHECK(cudaMemcpy(rhs_ptrs.data(), rhs.ptrs(), rhs.size()*sizeof(void*), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < batch_size; ++i) {
    if (lhs.sizes()[i] != rhs_sizes[i])
      return false;

    const uint8_t* lhs_ptr = reinterpret_cast<const uint8_t*>(lhs.ptrs()[i]);
    const uint8_t* rhs_ptr = reinterpret_cast<const uint8_t*>(rhs_ptrs[i]);
    std::vector<uint8_t> rhs_data(rhs_sizes[i]);
    CUDA_CHECK(cudaMemcpy(rhs_data.data(), rhs_ptr, rhs_sizes[i], cudaMemcpyDeviceToHost));

    for (size_t j = 0; j < rhs_sizes[i]; ++j)
      if (lhs_ptr[j] != rhs_data[j])
        return false;
  }

  return true;
}

// Benchmark performance from the binary data file fname
static void
run_example(const std::vector<std::vector<char>>& data)
{
  size_t total_bytes = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16;

  // build up input batch on CPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  std::cout << "chunks: " << input_data_cpu.size() << std::endl;

  // compression

  // Get max output size per chunk
  nvcompError_t status;
  size_t max_out_bytes;
  status = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
      chunk_size, &max_out_bytes);
  CHECK_NVCOMP_STATUS(status);

  // Allocate and prepare output/compressed batch
  BatchDataCPU compress_data_cpu(max_out_bytes, input_data_cpu.size());

#ifdef ENABLE_GDEFLATE
  // Compress on the CPU using gdeflate CPU batched API
  gdeflate::compressCPU(input_data_cpu.ptrs(),
      input_data_cpu.sizes(),
      chunk_size,
      input_data_cpu.size(),
      compress_data_cpu.ptrs(),
      compress_data_cpu.sizes());
#else
  throw std::runtime_error("nvcomp configured without gdeflate support. "
      "Please check the documentation for details on configuring nvcomp with gdeflate.")
#endif

  // compute compression ratio
  size_t* compressed_sizes_host = compress_data_cpu.sizes();
  size_t comp_bytes = 0;
  for (size_t i = 0; i < compress_data_cpu.size(); ++i)
    comp_bytes += compressed_sizes_host[i];

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // Copy compressed data to GPU
  BatchData compress_data(compress_data_cpu, true);

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(input_data_cpu, false);

  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // CUDA events to measure decompression time
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // gdeflate GPU decompression
  size_t decomp_temp_bytes;
  status = nvcompBatchedGdeflateDecompressGetTempSize(
      compress_data.size(), chunk_size, &decomp_temp_bytes);

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Run decompression
  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(),
      compress_data.sizes(),
      decomp_data.sizes(),
      chunk_size,
      compress_data.size(),
      d_decomp_temp,
      decomp_temp_bytes,
      decomp_data.ptrs(),
      stream);

  // Validate decompressed data against input
  if (!(input_data_cpu == decomp_data))
    throw std::runtime_error("Failed to validate decompressed data");
  else
    std::cout << "decompression validated :)" << std::endl;

  // Re-run decompression to get throughput
  cudaEventRecord(start, stream);
  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(),
      compress_data.sizes(),
      decomp_data.sizes(),
      chunk_size,
      compress_data.size(),
      d_decomp_temp,
      decomp_temp_bytes,
      decomp_data.ptrs(),
      stream);
  cudaEventRecord(end, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK_NVCOMP_STATUS(status);

  float ms;
  cudaEventElapsedTime(&ms, start, end);

  double decompression_throughput = ((double)total_bytes / ms) * 1e-6;
  std::cout << "decompression throughput (GB/s): "
            << decompression_throughput << std::endl;

  cudaFree(d_decomp_temp);

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaStreamDestroy(stream);
}
#undef CHECK_NVCOMP_STATUS

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

  run_example(data);

  return 0;
}
