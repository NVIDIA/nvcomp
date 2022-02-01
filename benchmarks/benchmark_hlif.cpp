/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

// Benchmark performance from the binary data file fname
#include <vector>
#include <string.h>

#include "benchmark_common.h"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

template<typename T = uint8_t>
void run_benchmark(const std::vector<T>& data, nvcompManagerBase& batch_manager, int verbose_memory, cudaStream_t stream)
{
  size_t input_element_count = data.size();

  // Make sure dataset fits on GPU to benchmark total compression
  size_t freeMem;
  size_t totalMem;
  CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
  if (freeMem < input_element_count * sizeof(T)) {
    std::cout << "Insufficient GPU memory to perform compression." << std::endl;
    exit(1);
  }
  
  std::cout << "----------" << std::endl;
  std::cout << "uncompressed (B): " << data.size() * sizeof(T) << std::endl;

  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input_element_count;
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, data.data(), in_bytes, cudaMemcpyHostToDevice));

  auto compress_config = batch_manager.configure_compression(in_bytes);
  
  size_t comp_out_bytes = compress_config.max_compressed_buffer_size;
  benchmark_assert(
      comp_out_bytes > 0, "Output size must be greater than zero.");

  // Allocate temp workspace
  size_t comp_scratch_bytes = batch_manager.get_required_scratch_buffer_size();
  uint8_t* d_comp_scratch;
  CUDA_CHECK(cudaMalloc(&d_comp_scratch, comp_scratch_bytes));
  batch_manager.set_scratch_buffer(d_comp_scratch);

  // Allocate compressed output buffer
  uint8_t* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  if (verbose_memory) {
    std::cout << "compression memory (input+output+scratch) (B): "
              << (in_bytes + comp_out_bytes + comp_scratch_bytes) << std::endl;
    std::cout << "compression scratch space (B): " << comp_scratch_bytes << std::endl;
    std::cout << "compression output space (B): " << comp_out_bytes
              << std::endl;
  }

  // Launch compression
  auto start = std::chrono::steady_clock::now();
  batch_manager.compress(
      d_in_data,
      d_comp_out,
      compress_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  comp_out_bytes = batch_manager.get_compressed_output_size(d_comp_out);

  CUDA_CHECK(cudaFree(d_comp_scratch));
  CUDA_CHECK(cudaFree(d_in_data));

  std::cout << "comp_size: " << comp_out_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)data.size() * sizeof(T) / comp_out_bytes << std::endl;
  std::cout << "compression throughput (GB/s): "
            << gbs(start, end, data.size() * sizeof(T)) << std::endl;

  // allocate output buffer
  auto decomp_config = batch_manager.configure_decompression(d_comp_out);
  const size_t decomp_bytes = decomp_config.decomp_data_size;
  uint8_t* decomp_out_ptr;
  CUDA_CHECK(cudaMalloc(&decomp_out_ptr, decomp_bytes));
  const size_t decomp_temp_bytes = 0;

  // get output size
  if (verbose_memory) {
    std::cout << "decompression memory (input+output+temp) (B): "
              << (decomp_bytes + comp_out_bytes + decomp_temp_bytes)
              << std::endl;
    std::cout << "decompression temp space (B): " << decomp_temp_bytes
              << std::endl;
  }

  start = std::chrono::steady_clock::now();

  // execute decompression (asynchronous)
  batch_manager.decompress(decomp_out_ptr, d_comp_out, decomp_config);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  end = std::chrono::steady_clock::now();

  std::cout << "decompression throughput (GB/s): "
            << gbs(start, end, decomp_bytes) << std::endl;

  CUDA_CHECK(cudaFree(d_comp_out));

  benchmark_assert(
      decomp_bytes == input_element_count * sizeof(T),
      "Decompressed result incorrect size.");

  std::vector<T> res(input_element_count);
  cudaMemcpy(
      res.data(),
      decomp_out_ptr,
      input_element_count * sizeof(T),
      cudaMemcpyDeviceToHost);
  
  CUDA_CHECK(cudaFree(decomp_out_ptr));
  
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

void run_benchmark_from_file(char* fname, nvcompManagerBase& batch_manager, int verbose_memory, cudaStream_t stream)
{
  using T = uint8_t;

  size_t input_elts = 0;
  std::vector<T> data;
  data = load_dataset_from_binary<T>(fname, &input_elts);
  run_benchmark(data, batch_manager, verbose_memory, stream);
}

static void print_usage()
{
  printf("Usage: benchmark_hlif [format_type] [OPTIONS]\n");
  printf("  %-35s One of <snappy / bitcomp / ans / cascaded/ gdeflate / lz4>\n", "[ format_type ]");
  printf("  %-35s Binary dataset filename (required).\n", "-f, --filename");
  printf("  %-35s Chunk size (default 64 kB).\n", "-c, --chunk-size");
  printf("  %-35s GPU device number (default 0)\n", "-g, --gpu");
  printf("  %-35s Data type (default 'char', options are 'char', 'short', 'int')\n", "-t, --type");
  printf(
      "  %-35s Output GPU memory allocation sizes (default off)\n",
      "-m --memory");
  exit(1);
}

int main(int argc, char* argv[])
{
  char* fname = NULL;
  int gpu_num = 0;
  int verbose_memory = 0;

  // Cascaded opts
  nvcompBatchedCascadedOpts_t cascaded_opts = nvcompBatchedCascadedDefaultOpts;

  // Shared opts
  int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  std::string comp_format;

  bool explicit_type = false;
  bool explicit_chunk_size = false;

  // Parse command-line arguments
  char** argv_end = argv + argc;
  argv += 1;

  // First the format
  comp_format = std::string{*argv++};
  if (comp_format == "lz4") {
  } else if (comp_format == "snappy") {
  } else if (comp_format == "bitcomp") {
  } else if (comp_format == "ans") {
  } else if (comp_format == "cascaded") {
  } else if (comp_format == "gdeflate") {
  } else {
    printf("invalid format\n");
    print_usage();
    return 1;
  }

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

    if (strcmp(arg, "--chunk-size") == 0 || strcmp(arg, "-c") == 0) {
      chunk_size = atoi(optarg);
      explicit_chunk_size = true;
      continue;
    }

    if (strcmp(arg, "--type") == 0 || strcmp(arg, "-t") == 0) {
      explicit_type = true;
      if (strcmp(optarg, "char") == 0) {
        data_type = NVCOMP_TYPE_CHAR;
      } else if (strcmp(optarg, "short") == 0) {
        data_type = NVCOMP_TYPE_SHORT;
      } else if (strcmp(optarg, "int") == 0) {
        data_type = NVCOMP_TYPE_INT;
      } else if (strcmp(optarg, "longlong") == 0) {
        data_type = NVCOMP_TYPE_LONGLONG;
      } else {
        print_usage();
        return 1;
      }
      continue;
    }

    if (strcmp(arg, "--num_rles") == 0 || strcmp(arg, "-r") == 0) {
      cascaded_opts.num_RLEs = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--num_deltas") == 0 || strcmp(arg, "-d") == 0) {
      cascaded_opts.num_deltas = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--num_bps") == 0 || strcmp(arg, "-b") == 0) {
      cascaded_opts.use_bp = (atoi(optarg) != 0);
      continue;
    }

    print_usage();
    return 1;
  }

  if (fname == NULL) {
    print_usage();
    return 1;
  }

  cudaSetDevice(gpu_num);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::unique_ptr<nvcompManagerBase> manager;
  if (comp_format == "lz4") {
    manager = std::make_unique<LZ4Manager>(chunk_size, data_type, stream);
  } else if (comp_format == "snappy") {
    manager = std::make_unique<SnappyManager>(chunk_size, stream);
  } else if (comp_format == "bitcomp") {
    manager = std::make_unique<BitcompManager>(data_type, 0 /* algo--fixed for now */, stream);
  } else if (comp_format == "ans") {
    manager = std::make_unique<ANSManager>(chunk_size, stream);
  } else if (comp_format == "cascaded") {
    if (explicit_type) {
      cascaded_opts.type = data_type;
    }

    if (explicit_chunk_size) {
      cascaded_opts.chunk_size = chunk_size;
    }

    manager = std::make_unique<CascadedManager>(cascaded_opts, stream);
  } else if (comp_format == "gdeflate") {
    manager = std::make_unique<GdeflateManager>(chunk_size, 0 /* algo--fixed for now */, stream);
  } else {
    print_usage();
    return 1;
  }

  run_benchmark_from_file(fname, *manager, verbose_memory, stream);
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}