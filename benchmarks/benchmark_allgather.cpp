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

#include <nvml.h>
#include "benchmark_common.h"
#include "lz4.hpp"
#include "cascaded.hpp"

#include <algorithm>
#include <getopt.h>
#include <iostream>

using namespace nvcomp;

#define MAX_STREAMS 8 // Max number of streams used by each GPU for compression/decompression

const char * convertToComputeModeString(nvmlComputeMode_t mode)
{
    switch (mode)
    {
        case NVML_COMPUTEMODE_DEFAULT:
            return "Default";
        case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
            return "Exclusive_Thread";
        case NVML_COMPUTEMODE_PROHIBITED:
            return "Prohibited";
        case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
            return "Exclusive Process";
        default:
            return "Unknown";
    }
}

void testNVML()
{
    nvmlReturn_t result;
    unsigned int device_count, i;

    // First initialize NVML library
    result = nvmlInit();
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));

        printf("Press ENTER to continue...\n");
        getchar();
        return;
    }

    result = nvmlDeviceGetCount(&device_count);
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to query device count: %s\n", nvmlErrorString(result));
        return;
    }
    printf("Found %d device%s\n\n", device_count, device_count != 1 ? "s" : "");

    printf("Listing devices:\n");    
    for (i = 0; i < device_count; i++)
    {
        nvmlDevice_t device;
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlPciInfo_t pci;
        nvmlComputeMode_t compute_mode;

        // Query for device handle to perform operations on a device
        // You can also query device handle by other features like:
        // nvmlDeviceGetHandleBySerial
        // nvmlDeviceGetHandleByPciBusId
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
            return;
        }

        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get name of device %i: %s\n", i, nvmlErrorString(result));
            return;
        }
        
        // pci.busId is very useful to know which device physically you're talking to
        // Using PCI identifier you can also match nvmlDevice handle to CUDA device.
        result = nvmlDeviceGetPciInfo(device, &pci);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get pci info for device %i: %s\n", i, nvmlErrorString(result));
            return;
        }

        printf("%d. %s [%s]\n", i, name, pci.busId);

        // This is a simple example on how you can modify GPU's state
        result = nvmlDeviceGetComputeMode(device, &compute_mode);
        if (NVML_ERROR_NOT_SUPPORTED == result)
            printf("\t This is not CUDA capable device\n");
        else if (NVML_SUCCESS != result)
        { 
            printf("Failed to get compute mode for device %i: %s\n", i, nvmlErrorString(result));
            return;
        }
        else
        {
            // try to change compute mode
            printf("\t Changing device's compute mode from '%s' to '%s'\n", 
                    convertToComputeModeString(compute_mode), 
                    convertToComputeModeString(NVML_COMPUTEMODE_PROHIBITED));

            result = nvmlDeviceSetComputeMode(device, NVML_COMPUTEMODE_PROHIBITED);
            if (NVML_ERROR_NO_PERMISSION == result)
                printf("\t\t Need root privileges to do that: %s\n", nvmlErrorString(result));
            else if (NVML_ERROR_NOT_SUPPORTED == result)
                printf("\t\t Compute mode prohibited not supported. You might be running on\n"
                       "\t\t windows in WDDM driver model or on non-CUDA capable GPU.\n");
            else if (NVML_SUCCESS != result)
            {
                printf("\t\t Failed to set compute mode for device %i: %s\n", i, nvmlErrorString(result));
                return;
            } 
            else
            {
                printf("\t Restoring device's compute mode back to '%s'\n", 
                        convertToComputeModeString(compute_mode));
                result = nvmlDeviceSetComputeMode(device, compute_mode);
                if (NVML_SUCCESS != result)
                { 
                    printf("\t\t Failed to restore compute mode for device %i: %s\n", i, nvmlErrorString(result));
                    return;
                }
            }
        }
    }
}


static void print_usage()
{
  printf("Usage: benchmark_binary [OPTIONS]\n");
  printf("  %-35s Binary dataset filename (required).\n", "-f, --filename");
  printf("  %-35s Number of GPUs (default 2)\n", "-g, --gpu");
  printf("  %-35s Number of data chunks (default number of GPUs)\n", "-h, --chunks");
  printf("  %-35s Compression type (None, Cascaded, LZ4)\n", "-c, --compression");
  printf("  %-35s *If Cascaded* Number lf RLEs (default 1)\n", "-r, --rles");
  printf("  %-35s *If Cascaded* Number of Deltas (default 0)\n", "-d, --deltas");
  printf("  %-35s *If Cascaded* Bitpacking enabled (default 1)\n", "-b, --bitpack");
  printf("  %-35s *If Cascaded* Datatype (int or long, default int)\n", "-t, --type");
  exit(1);
}


// Check that the output of the benchmark matches the input
template<typename T>
static void check_output(std::vector<T**>& outputs, T* h_data, int gpus, int chunks, size_t* chunk_sizes) 
{
  T* result_buffer = new T[chunk_sizes[0]];
  size_t idx;

  for(int i=0; i<gpus; ++i) {
    idx=0; 
    for(int chunkId=0; chunkId < chunks; ++chunkId) {
      CUDA_CHECK(cudaMemcpy(result_buffer, outputs[i][chunkId], chunk_sizes[chunkId]*sizeof(T), cudaMemcpyDeviceToHost));
      for(size_t j=0; j<chunk_sizes[chunkId]; ++j) {
        if(result_buffer[j] != h_data[idx+j]) {
          std::cout << "Incorrect result, elemenet number:" << j 
            << " - expected:" << h_data[j] << ", found:" << result_buffer[j] << std::endl;
          exit(1);
        }
      } 
      idx+=chunk_sizes[chunkId];
    }
  }
  delete result_buffer;
}
    

// Load the input from a binary file, split it unti chunks, and distribute them to the GPU to prepare for benchmark
template <typename T>
static void load_chunks_to_devices(
    char* fname,
    int binary_file,
    int gpus,
    int chunks,
    T** dev_ptrs,
    size_t* data_sizes,
    std::vector<T>* h_data)
{

  size_t input_elts=0;
  if (binary_file == 0) {
    *h_data = load_dataset_from_txt<T>(fname, &input_elts);
  } else {
    *h_data = load_dataset_from_binary<T>(fname, &input_elts);
  }

  int chunks_per_gpu = chunks/gpus;
  if(chunks_per_gpu*gpus != chunks) {
    std::cout << "chunks does not evenly divide number of GPUs." << std::endl;
    exit(1);
  }

  int chunk_size = 1 + ((input_elts-1) / chunks);

  std::cout << "Loading data - " << input_elts << " elements, chunk size:" << chunk_size << ", total chunks:" << chunks
            << ", number of gpus being used:" << gpus << std::endl;

  size_t offset;

  for(int i=0; i<gpus; i++) {
    cudaSetDevice(i);

    // Make sure dataset fits on GPU to benchmark total compression
    size_t freeMem;
    size_t totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (freeMem < chunk_size * chunks_per_gpu * sizeof(T)) {
      std::cout << "Insufficient GPU memory to perform compression." << std::endl;
      exit(1);
    }

    for(int j=0; j<chunks_per_gpu; j++) {
      offset = i*chunks_per_gpu + j;
      data_sizes[offset] = chunk_size;
      if(((size_t)(offset+1)*chunk_size) > h_data->size()) {
        data_sizes[offset] = h_data->size() - (offset*chunk_size);
      }
      CUDA_CHECK(cudaMalloc(&dev_ptrs[offset], data_sizes[offset]*sizeof(T)));
      CUDA_CHECK(cudaMemcpy(dev_ptrs[offset], &((h_data->data())[(offset)*chunk_size]), data_sizes[offset]*sizeof(T), cudaMemcpyHostToDevice));
    }
  }
}

// Perform the all-gather operation using memory copies between GPUs
template<typename T>
void copy_to_all(int gpus, int chunks, T** dev_ptrs, size_t* chunk_bytes, T*** dest_ptrs, cudaStream_t** streams, int STREAMS_PER_GPU) {
//  int chunks_per_gpu = chunks/gpus;

  for(int i=0; i<chunks; ++i) {
    for(int j=0; j<gpus; ++j) {
      if (i != j) {
        CUDA_CHECK(cudaMemcpyAsync(dest_ptrs[j][i], dev_ptrs[i], chunk_bytes[i], cudaMemcpyDeviceToDevice, streams[j][i%STREAMS_PER_GPU]));
      }
    }
  } 
}

// Create all needed streams one each GPU
static void create_gpu_streams(std::vector<cudaStream_t*>* streams, const int gpus, int STREAMS_PER_GPU) {
  streams->reserve(gpus);
  for(int i=0; i<gpus; ++i) {
    cudaSetDevice(i);
    streams->push_back( new cudaStream_t[STREAMS_PER_GPU]);
    for(int j=0; j<STREAMS_PER_GPU; ++j) {
      cudaStreamCreateWithFlags(&((*streams)[i][j]), cudaStreamNonBlocking);
    }
  }
}

// Synchronize all streams on all GPUs
static void sync_all_streams(std::vector<cudaStream_t*>* streams, int gpus, int STREAMS_PER_GPU) {
  for(int i=0; i<gpus; ++i) {
    for(int j=0; j<STREAMS_PER_GPU; ++j) {
      cudaStreamSynchronize((*streams)[i][j]);
    }
  }
}

// Benchmark the All-gather operation on totally uncompressed data
template <typename T>
static void run_uncompressed_benchmark(
    const int gpus,
    const int chunks,
    T** dev_ptrs,
    size_t* chunk_sizes,
    std::vector<T>* h_data)
{

  const int chunks_per_gpu = chunks/gpus;
  const int STREAMS_PER_GPU = std::min(chunks_per_gpu, MAX_STREAMS);

  std::vector<T**> dest_ptrs;
  dest_ptrs.reserve(gpus);

  std::vector<cudaStream_t*> streams;
  create_gpu_streams(&streams, gpus, STREAMS_PER_GPU);

  for(int i=0; i<gpus; ++i) { // Allocate full data size on each GPU
    cudaSetDevice(i);
    dest_ptrs.push_back(new T*[chunks]);
    for(int j=0; j<chunks; ++j) {
      CUDA_CHECK(cudaMalloc(&dest_ptrs[i][j], chunk_sizes[j]*sizeof(T)));
    }
  }
  
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  copy_to_all<T>(gpus, chunks, dev_ptrs, chunk_sizes, dest_ptrs.data(), streams.data(), STREAMS_PER_GPU);
  for (int gpu = 0; gpu < gpus; ++gpu) {
    CUDA_CHECK(cudaMemcpyAsync(dest_ptrs[gpu][gpu], dev_ptrs[gpu], chunk_sizes[gpu]*sizeof(T), cudaMemcpyDeviceToDevice, streams[gpu][gpu%STREAMS_PER_GPU]));
  }

  sync_all_streams(&streams, gpus, STREAMS_PER_GPU);

  
  clock_gettime(CLOCK_MONOTONIC, &end);
  std::cout << "Full data size (B): " << h_data->size()*sizeof(T) << std::endl 
           << "Per-GPU benchmark throughput (GB/s): " << gbs(start, end, h_data->size()*(((double)gpus-1.0)/(double)gpus)*sizeof(T)) << std::endl;
  std::cout << "Total data transferred across system (B): " << h_data->size()*(gpus-1)*sizeof(T) << std::endl
           << "Total system throughput (GB/s): " << gbs(start, end, h_data->size()*(gpus-1)*sizeof(T)) << std::endl;


// Test for correctness
  check_output<T>(dest_ptrs, h_data->data(), gpus, chunks, chunk_sizes);

}

// Benchmark the performance of the All-gather operation using LZ4 compression/decompression to reduce data transfers
static void run_lz4_benchmark(
    const int gpus,
    const int chunks,
    uint8_t** dev_ptrs,
    size_t* chunk_sizes,
    std::vector<uint8_t>* h_data)
{
  using T = uint8_t;

  const int chunks_per_gpu = chunks/gpus;
  const int STREAMS_PER_GPU = std::min(chunks_per_gpu, MAX_STREAMS);

  std::vector<T**> dest_ptrs;
  dest_ptrs.reserve(gpus);
  int idx;
  size_t total_comp_bytes=0;
 
  for(int i=0; i<gpus; ++i) { // Allocate full data size on each GPU
    cudaSetDevice(i);
    dest_ptrs.push_back(new T*[chunks]);
    for(int j=0; j<chunks; ++j) {
      CUDA_CHECK(cudaMalloc(&dest_ptrs[i][j], chunk_sizes[j]*sizeof(T)));
    }
  }

// Create a compressor for each chunk
  LZ4Compressor<T>** compressors = new LZ4Compressor<T>*[chunks];
  size_t* comp_out_bytes;
  CUDA_CHECK(cudaMallocHost(&comp_out_bytes, chunks*sizeof(size_t)));
  std::vector<void*> d_comp_out(chunks);

  std::vector<cudaStream_t*> streams;
  create_gpu_streams(&streams, gpus, STREAMS_PER_GPU);

// Create temp buffers for each GPU to use for compression and decompression
  std::vector<size_t> temp_bytes;
  temp_bytes.reserve(gpus);
  std::vector<T**> d_temp;
  d_temp.reserve(gpus);

// Allocate all memory buffers necessary for compression of each chunk
  for(int gpu=0; gpu < gpus; ++gpu) {
    cudaSetDevice(gpu);

    temp_bytes[gpu] = 0;
// Create compressor each chunk
    for(int chunkIdx=0; chunkIdx<chunks_per_gpu; ++chunkIdx) {
      idx = gpu*chunks_per_gpu+chunkIdx;
      compressors[idx] = new LZ4Compressor<T>(dev_ptrs[idx], chunk_sizes[idx], 1 << 16);

// Find largest temp buffer needed for any chunk
      if(compressors[idx]->get_temp_size() > temp_bytes[gpu]) {
        temp_bytes[gpu] = compressors[idx]->get_temp_size();
      }
    }

// Use one temp buffer for each stream on each gpu
    d_temp.push_back(new T*[STREAMS_PER_GPU]);
    for(int j=0; j<STREAMS_PER_GPU; ++j) {
      CUDA_CHECK(cudaMalloc(&d_temp[gpu][j], temp_bytes[gpu]));
    }

// Allocate output buffers for each chunk on the GPU
    for(int chunkIdx=0; chunkIdx<chunks_per_gpu; ++chunkIdx) {
      idx = gpu*chunks_per_gpu+chunkIdx;
      comp_out_bytes[idx] = compressors[idx]->get_max_output_size(d_temp[gpu][0], temp_bytes[gpu]);
      CUDA_CHECK(cudaMalloc(&d_comp_out[idx], comp_out_bytes[idx]));
      total_comp_bytes += comp_out_bytes[idx];
    }
  }

// Allocate all memory buffers for decompression
  std::vector<size_t> decomp_out_bytes;
  decomp_out_bytes.reserve(chunks*gpus);
  std::vector<T**> d_decomp_out; // output buffers for each chunk on each gpu
  d_decomp_out.reserve(gpus);
  for(int gpu=0; gpu < gpus; ++gpu) {
    d_decomp_out.push_back(new T*[chunks]);
    cudaSetDevice(gpu);
    for(int chunkId=0; chunkId<chunks; ++chunkId) {
      CUDA_CHECK(cudaMalloc(&d_decomp_out[gpu][chunkId], chunk_sizes[chunkId]*sizeof(T)));
    }
  }

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

// Issue compression calls
  for(int gpu=0; gpu<gpus; ++gpu) {
    cudaSetDevice(gpu);
    for(int chunkIdx=0; chunkIdx < chunks_per_gpu; ++chunkIdx) {
      idx = gpu*chunks_per_gpu+chunkIdx;
      compressors[idx]->compress_async(d_temp[gpu][chunkIdx%STREAMS_PER_GPU], temp_bytes[gpu], d_comp_out[idx], &comp_out_bytes[idx], streams[gpu][chunkIdx%STREAMS_PER_GPU]);
    }
  }

  sync_all_streams(&streams, gpus, STREAMS_PER_GPU);

  total_comp_bytes=0;
  for(int i=0; i<gpus*chunks_per_gpu; ++i) {
    total_comp_bytes += comp_out_bytes[i];
  }

//  Copy compressed data to all GPUs
  copy_to_all<T>(gpus, chunks, (T**)(d_comp_out.data()), comp_out_bytes, dest_ptrs.data(), streams.data(), STREAMS_PER_GPU);
  for (int gpu = 0; gpu < gpus; ++gpu) {
    CUDA_CHECK(cudaMemcpyAsync(d_decomp_out[gpu][gpu], dev_ptrs[gpu], chunk_sizes[gpu]*sizeof(T), cudaMemcpyDeviceToDevice, streams[gpu][gpu%STREAMS_PER_GPU]));
  }

// Create decompressors for each chunk on each gpu
  Decompressor<T>** decompressors = new Decompressor<T>*[chunks*gpus];

  for(int gpu=0; gpu < gpus; ++gpu) {
    cudaSetDevice(gpu);
    for(int chunkIdx=0; chunkIdx<chunks; ++chunkIdx) {
      idx = gpu*chunks+chunkIdx;
// Create compressor for the chunk and allocate necessary memory
      if (chunkIdx != gpu) {
        decompressors[idx] = new Decompressor<T>(dest_ptrs[gpu][chunkIdx], comp_out_bytes[chunkIdx], streams[gpu][chunkIdx%STREAMS_PER_GPU]);
        decomp_out_bytes[idx] = decompressors[idx]->get_output_size();

  // Check that temp space is sufficient
        if(temp_bytes[gpu] < decompressors[idx]->get_temp_size()) {
          std::cout << "Insufficient temp storage - size:" << temp_bytes[gpu] << ", needed:" << decompressors[idx]->get_temp_size()
                   << std::endl;
          exit(1);
        }
      } else {
        decomp_out_bytes[idx] = chunk_sizes[idx];
      }
    }
  }

// Issue decompression
  for(int gpu=0; gpu < gpus; ++gpu) {
    cudaSetDevice(gpu);
    for(int chunkIdx=0; chunkIdx < chunks; ++chunkIdx) {
      if (chunkIdx != gpu) {
        idx = gpu*chunks+chunkIdx;
        decompressors[idx]->decompress_async(d_temp[gpu][chunkIdx%STREAMS_PER_GPU], temp_bytes[gpu], d_decomp_out[gpu][chunkIdx], decomp_out_bytes[idx], streams[gpu][chunkIdx%STREAMS_PER_GPU]);
      }
    }
  }

  sync_all_streams(&streams, gpus, STREAMS_PER_GPU);

  clock_gettime(CLOCK_MONOTONIC, &end);

// Test for correctness
  check_output<T>(d_decomp_out, h_data->data(), gpus, chunks, chunk_sizes);

// Clean up
  for(int i=0; i<gpus; ++i) {
    for(int j=0; j<chunks; ++j) {
      CUDA_CHECK(cudaFree(d_decomp_out[i][j]));
      CUDA_CHECK(cudaFree(dest_ptrs[i][j]));
    }
    delete d_decomp_out[i];
    delete dest_ptrs[i];
    for(int j=0; j<STREAMS_PER_GPU; ++j) {
      CUDA_CHECK(cudaFree(d_temp[i][j]));
    }
    delete d_temp[i];
    delete streams[i];
  }
  for(int i=0; i<chunks; i++) {
    delete compressors[i];
  }
  delete compressors;
  for(int i=0; i<gpus; i++) {
    for(int j=0; j<chunks; ++j) {
      if (i != j) {
        delete decompressors[i*chunks+j];
      }
    }
  }
  delete decompressors;
  CUDA_CHECK(cudaFreeHost(comp_out_bytes));
  
  std::cout << "Full data size (B): " << h_data->size()*sizeof(T) << std::endl
           << "Per-GPU benchmark throughput (GB/s): " << gbs(start, end, h_data->size()*(((double)gpus-1.0)/(double)gpus)*sizeof(T)) << std::endl;
  std::cout << "Compressed data size (B): " << total_comp_bytes << ", compression ratio: "
            << (double)h_data->size()*sizeof(T) / (double)total_comp_bytes << std::endl;
  std::cout << "Total data distributed across system (B): " << h_data->size()*(gpus-1)*sizeof(T) << std::endl
           << "Total system throughput (GB/s): " << gbs(start, end, h_data->size()*(gpus-1)*sizeof(T)) << std::endl;
}


// Benchmark the performance of the All-gather operation using LZ4 compression/decompression to reduce data transfers
template<typename T>
static void run_cascaded_benchmark(
    int gpus,
    int chunks,
    T** dev_ptrs,
    size_t* chunk_sizes,
    std::vector<T>* h_data,
    int RLEs,
    int deltas,
    int bitPacking)
{

  const int chunks_per_gpu = chunks/gpus;
  const int STREAMS_PER_GPU = std::min(chunks_per_gpu, MAX_STREAMS);

  std::vector<T**> dest_ptrs;
  dest_ptrs.reserve(gpus);
  int idx=0;
  int total_comp_bytes=0;
 
// Create a compressor for each chunk
  CascadedCompressor<T>** compressors = new CascadedCompressor<T>*[chunks];
  std::vector<size_t> temp_bytes(gpus);
  std::vector<T**> d_temp;
  d_temp.reserve(gpus);
  size_t* comp_out_bytes;
  CUDA_CHECK(cudaMallocHost(&comp_out_bytes, chunks*sizeof(size_t)));
  std::vector<void*> d_comp_out(chunks);

  std::vector<cudaStream_t*> streams;

  create_gpu_streams(&streams, gpus, STREAMS_PER_GPU);

// Allocate all memory buffers necessary for compression of each chunk
  for(int gpu=0; gpu < gpus; ++gpu) {
    cudaSetDevice(gpu);

// Create compressor for the chunk and allocate necessary memory
    for(int chunkIdx=0; chunkIdx<chunks_per_gpu; ++chunkIdx) {
      idx = gpu*chunks_per_gpu+chunkIdx;
      compressors[idx] = new CascadedCompressor<T>(dev_ptrs[idx], chunk_sizes[idx], RLEs, deltas, bitPacking);
    }

// Use one temp buffer for each stream on each gpu
//    streams.push_back(new cudaStream_t[STREAMS_PER_GPU]);
    d_temp.push_back(new T*[STREAMS_PER_GPU]);

// biggest temp buffer requirement
    temp_bytes[gpu] = 0;
    for(int j=0; j<chunks_per_gpu; ++j) {
      const size_t req_bytes = std::max(5*chunk_sizes[gpu*chunks_per_gpu+j]*sizeof(T), compressors[gpu*chunks_per_gpu+j]->get_temp_size());
      if(temp_bytes[gpu] < req_bytes) {
        temp_bytes[gpu] = req_bytes;
      }
    }
      
    for(int j=0; j<STREAMS_PER_GPU; ++j) {
//      cudaStreamCreateWithFlags(&streams[gpu][j], cudaStreamNonBlocking);
      CUDA_CHECK(cudaMalloc(&d_temp[gpu][j], temp_bytes[gpu]));
    }

// Allocate output buffers for each chunk on the GPU
    for(int chunkIdx=0; chunkIdx<chunks_per_gpu; ++chunkIdx) {
      idx = gpu*chunks_per_gpu+chunkIdx;
      comp_out_bytes[idx] = compressors[idx]->get_max_output_size(
          d_temp[gpu][0], temp_bytes[gpu]);
      CUDA_CHECK(cudaMalloc(&d_comp_out[idx], comp_out_bytes[idx]));
      total_comp_bytes += comp_out_bytes[idx];
    }
  }

  for(int i=0; i<gpus; ++i) { // Allocate full data size on each GPU
    cudaSetDevice(i);
    dest_ptrs.push_back(new T*[chunks]);
    for(int j=0; j<chunks; ++j) {
      CUDA_CHECK(cudaMalloc(&dest_ptrs[i][j], comp_out_bytes[idx]));
    }
  }

// Allocate all memory buffers for decompression
  std::vector<size_t> decomp_out_bytes;
  decomp_out_bytes.resize(chunks*gpus);
  std::vector<T**> d_decomp_out; // output buffers for each chunk on each gpu
  d_decomp_out.reserve(gpus);
  for(int gpu=0; gpu < gpus; ++gpu) {
    d_decomp_out.push_back(new T*[chunks]);

    cudaSetDevice(gpu);
    for(int chunkId=0; chunkId<chunks; ++chunkId) {
      CUDA_CHECK(cudaMalloc(&d_decomp_out[gpu][chunkId], chunk_sizes[chunkId]*sizeof(T)));
    }
  }

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

// Issue compression calls
  for(int gpu=0; gpu<gpus; ++gpu) {
    cudaSetDevice(gpu);
    for(int chunkIdx=0; chunkIdx < chunks_per_gpu; ++chunkIdx) {
      idx = gpu*chunks_per_gpu+chunkIdx;
      compressors[idx]->compress_async(d_temp[gpu][chunkIdx%STREAMS_PER_GPU], temp_bytes[gpu], d_comp_out[idx], &comp_out_bytes[idx], streams[gpu][chunkIdx%STREAMS_PER_GPU]);
    }
  }

  sync_all_streams(&streams, gpus, STREAMS_PER_GPU);

  total_comp_bytes=0;
  for(int i=0; i<gpus*chunks_per_gpu; ++i) {
    total_comp_bytes += comp_out_bytes[i];
  }

//  Copy compressed data to all GPUs
  copy_to_all<T>(gpus, chunks, (T**)(d_comp_out.data()), comp_out_bytes, dest_ptrs.data(), streams.data(), STREAMS_PER_GPU);
  for (int gpu = 0; gpu < gpus; ++gpu) {
    CUDA_CHECK(cudaMemcpyAsync(d_decomp_out[gpu][gpu], dev_ptrs[gpu], chunk_sizes[gpu]*sizeof(T), cudaMemcpyDeviceToDevice, streams[gpu][gpu%STREAMS_PER_GPU]));
  }

// Create decompressors for each chunk on each gpu
  Decompressor<T>** decompressors = new Decompressor<T>*[chunks*gpus];

  for(int gpu=0; gpu < gpus; ++gpu) {
    cudaSetDevice(gpu);
    for(int chunkIdx=0; chunkIdx<chunks; ++chunkIdx) {
      if (gpu != chunkIdx) {
        idx = gpu*chunks+chunkIdx;
  // Create compressor for the chunk and allocate necessary memory
        decompressors[idx] = new Decompressor<T>(dest_ptrs[gpu][chunkIdx], comp_out_bytes[chunkIdx], streams[gpu][chunkIdx%STREAMS_PER_GPU]);
        decomp_out_bytes[idx] = decompressors[idx]->get_output_size();
      } else {
        decomp_out_bytes[idx] = chunk_sizes[chunkIdx]*sizeof(T);
      }
    }

// find biggest temp buffer requirement
    for(int j=0; j<chunks; ++j) {
      if(j != gpu && temp_bytes[gpu] < decompressors[gpu*chunks+j]->get_temp_size()) {
        std::cout << "Insufficient temp storage size for gpu " << gpu << ", chunk " << j << ": " << temp_bytes[gpu] << ", needed:" << decompressors[idx]->get_temp_size()
                 << std::endl;
        exit(1);
      }
    }
  }

// Issue decompression calls
  for(int gpu=0; gpu < gpus; ++gpu) {
    cudaSetDevice(gpu);
    for(int chunkIdx=0; chunkIdx < chunks; ++chunkIdx) {
      if (gpu != chunkIdx) {
        idx = gpu*chunks+chunkIdx;
        decompressors[idx]->decompress_async(d_temp[gpu][chunkIdx%STREAMS_PER_GPU], temp_bytes[gpu], d_decomp_out[gpu][chunkIdx], decomp_out_bytes[idx], streams[gpu][chunkIdx%STREAMS_PER_GPU]);
      }
    }
  }

  sync_all_streams(&streams, gpus, STREAMS_PER_GPU);

  clock_gettime(CLOCK_MONOTONIC, &end);

// Test for correctness
  check_output<T>(d_decomp_out, h_data->data(), gpus, chunks, chunk_sizes);

// Cleanup
  for(int i=0; i<gpus; ++i) {
    for(int j=0; j<chunks; ++j) {
      CUDA_CHECK(cudaFree(d_decomp_out[i][j]));
      CUDA_CHECK(cudaFree(dest_ptrs[i][j]));
    }
    delete d_decomp_out[i];
    delete dest_ptrs[i];
    for(int j=0; j<STREAMS_PER_GPU; ++j) {
      if (j != i) {
        CUDA_CHECK(cudaFree(d_temp[i][j]));
      }
    }
    delete d_temp[i];
    delete streams[i];
  }
  for(int i=0; i<chunks; i++) {
    delete compressors[i];
  }
  delete compressors;
  for(int i=0; i<gpus; ++i) {
    for(int j=0; j<chunks; ++j) {
      if (j != i) {
        delete decompressors[i*chunks+j];
      }
    }
  }
  delete decompressors;
  CUDA_CHECK(cudaFreeHost(comp_out_bytes));
  
  std::cout << "Full data size (B): " << h_data->size()*sizeof(T) << std::endl
           << "Per-GPU benchmark throughput (GB/s): " << gbs(start, end, h_data->size()*(((double)gpus-1.0)/(double)gpus)*sizeof(T)) << std::endl;
  std::cout << "Compressed data size (B): " << total_comp_bytes << ", compression ratio: "
            << (double)h_data->size()*sizeof(T) / (double)total_comp_bytes << std::endl;
  std::cout << "Total data distributed across system (B): " << h_data->size()*(gpus-1)*sizeof(T) << std::endl
           << "Total system throughput (GB/s): " << gbs(start, end, h_data->size()*(gpus-1)*sizeof(T)) << std::endl;
}

static void enable_nvlink(int gpus) {
  for(int i=0; i<gpus; ++i) {
    for(int j=0; j<gpus; ++j) {
      int can_access_A=0;
      cudaDeviceCanAccessPeer(&can_access_A, i, j);
      if(can_access_A) {
        cudaSetDevice(i);
        cudaDeviceEnablePeerAccess(j, 0);
      }
    }
  }
}


int main(int argc, char* argv[])
{
  char* fname = NULL;
  int gpu_num = 2;
  int chunks = 0;
  std::string comp_type = "none";
  int RLEs = 1;
  int deltas = 0;
  int bitPacking = 1;
  std::string dtype = "int";
  int binary_file = 1;

  // Parse command-line arguments
  while (1) {
    int option_index = 0;
    static struct option long_options[]{{"file", required_argument, 0, 'f'},
                                        {"gpu", required_argument, 0, 'g'},
                                        {"chunks", required_argument, 0, 'h'},
                                        {"comp", required_argument, 0, 'c'},
                                        {"rles", required_argument, 0, 'r'},
                                        {"deltas", required_argument, 0, 'd'},
                                        {"bitpack", required_argument, 0, 'b'},
                                        {"type", required_argument, 0, 't'},
                                        {"help", no_argument, 0, '?'}};
    int c;
    opterr = 0;
    c = getopt_long(
        argc, argv, "f:g:h:c:r:d:b:t:?", long_options, &option_index);
    if (c == -1)
      break;
    switch (c) {
    case 'f':
      if (startsWith(optarg, "TXT:")) {
        binary_file = 0;
        optarg = optarg + 4;
      }
      fname = optarg;
      break;
    case 'g':
      gpu_num = atoi(optarg);
      break;
    case 'h':
      chunks = atoi(optarg);
      break;
    case 'c':
      comp_type = optarg;
      break;
    case 'r':
      RLEs = atoi(optarg);
      break;
    case 'd':
      deltas = atoi(optarg);
      break;
    case 'b':
      bitPacking = atoi(optarg);
      break;
    case 't':
      dtype = optarg;
      break;
    case '?':
    default:
      print_usage();
      exit(1);
    }
  }
  if (fname == NULL) {
    print_usage();
  }

  if(chunks == 0) chunks = gpu_num;
 
  if(comp_type == "lz4" || comp_type == "LZ4" || comp_type == "none" || comp_type == "None") {
    dtype = "uint8"; // LZ4 only works on byte-level
  }

/*
nvmlReturn_t result;

char uuid[100];
char name[NVML_DEVICE_NAME_BUFFER_SIZE];
nvmlDevice_t dev_ptr;
for(int i=0; i<gpu_num; i++) {
  result = nvmlDeviceGetHandleByIndex(i, &dev_ptr);
  printf("getHand:%s\n", nvmlErrorString(result));
  nvmlDeviceGetUUID(dev_ptr, uuid, 100);
  nvmlDeviceGetName(dev_ptr, name, NVML_DEVICE_NAME_BUFFER_SIZE);
  printf("i:%d, uuid:%s, name:%s\n", i, uuid, name);
  std::cout << "i:" << i << ", UUID:" << uuid << ", name:" << name << std::endl;
}
*/

//  testNVML();

  enable_nvlink(gpu_num);

  if(dtype == "int") {
    std::vector<int32_t*> data_ptrs(chunks);
    std::vector<size_t> data_sizes(chunks);
    std::vector<int32_t> h_data;
    load_chunks_to_devices<int32_t>(fname, binary_file, gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data);
    run_cascaded_benchmark<int32_t>(gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data, RLEs, deltas, bitPacking);
  }
  else if(dtype == "long") {
    std::vector<int64_t*> data_ptrs(chunks);
    std::vector<size_t> data_sizes(chunks);
    std::vector<int64_t> h_data;
    load_chunks_to_devices<int64_t>(fname, binary_file, gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data);
    run_cascaded_benchmark<int64_t>(gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data, RLEs, deltas, bitPacking);
  }
  else if(dtype == "int8") {
    std::vector<int8_t*> data_ptrs(chunks);
    std::vector<size_t> data_sizes(chunks);
    std::vector<int8_t> h_data;
    load_chunks_to_devices<int8_t>(fname, binary_file, gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data);
    run_cascaded_benchmark<int8_t>(gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data, RLEs, deltas, bitPacking);
  }
  else if(dtype == "byte" || dtype == "uint8") {
    std::vector<uint8_t*> data_ptrs(chunks);
    std::vector<size_t> data_sizes(chunks);
    std::vector<uint8_t> h_data;
    load_chunks_to_devices<uint8_t>(fname, binary_file, gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data);

    if(comp_type == "lz4" || comp_type == "LZ4") {
// Run LZ4 benchmark
      run_lz4_benchmark(gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data);
    }
    else if(comp_type == "none" || comp_type == "None") {
// Run no-comp benchmark
      run_uncompressed_benchmark<uint8_t>(gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data);
    } 
    else {
      std::cout << "Invalid compression benchmark selected." << std::endl;
      print_usage();
      exit(1); 
    }
  }
  else {
    std::cout << "Invalid datatype selected." << std::endl;
    print_usage();
    exit(1); 
  }

  return 0;
}
