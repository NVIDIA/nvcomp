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

// Simple example of how to use GDS with nvcomp.
// GDS (GPU Direct Storage) allows to read and write from/to NVMe drives
// directly from the GPU, bypassing the CPU.
//
// For best performance, the I/O buffer should be registered but it is not
// mandatory. Registration can be expensive but done only once, and allows the
// I/O to be performed directly from the registered buffer. Otherwise, GDS will
// use it's own intermediate buffer, at the expense of extra memory copies.
// Similarly, I/Os with a base address or size which is not aligned on 4KB will
// go through GDS's internal buffer and will be less efficient.
//
// For more details on GDS, included the supported GPUs, please see the
// documentation. https://docs.nvidia.com/gpudirect-storage/
//
// To compile this GDS example, GDS must be installed, and the following
// option must be passed when configuring cmake:
// cmake -DBUILD_GDS_EXAMPLE=on <...>

#include <fcntl.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cufile.h>
#include <nvtx3/nvToolsExt.h>

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
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

// Kernel to initialize the input data with sequential bytes
__global__ void initialize(uint8_t* data, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    data[i] = i & 0xff;
}

// Kernel to compare 2 buffers. Invalid flag must be set to zero before.
__global__ void
compare(const uint8_t* ref, const uint8_t* val, int* invalid, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  while (i < n) {
    if (ref[i] != val[i])
      *invalid = 1;
    i += stride;
  }
}

void usage(const char* str)
{
  printf("Argument: %s <filename>\n", str);
  exit(-1);
}

int main(int argc, char** argv)
{
  if (argc != 2)
    usage(argv[0]);

  // Open the file. Note: GDS requires O_DIRECT.
  const char* filename = argv[1];
  int fd = open(filename, O_RDWR | O_TRUNC | O_CREAT | O_DIRECT, 0666);
  if (fd == -1) {
    printf("Error, cannot create the file: %s\n", filename);
    return -1;
  }
  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
  printf("Using device: %s\n", deviceProp.name);
  int smcount = deviceProp.multiProcessorCount;

  // Uncompressed data = 100 MB
  const size_t n = 100000000;

  // Device pointers for the data to be compressed / decompressed
  uint8_t *d_input, *d_output;
  cudaStream_t stream;
  CUDA_CHECK(cudaMalloc(&d_input, n));
  CUDA_CHECK(cudaMalloc(&d_output, n));
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Initialize the input data (sequential bytes)
  initialize<<<(n - 1) / 512 + 1, 512, 0, stream>>>(d_input, n);

  // Using NVTX to highlight the different phases of the program in the Nsight
  // Systems profiler
  nvtxRangePushA("Compressor setup");

  // Create an LZ4 compressor, get the max output size, and temp storage size
  LZ4Manager compressor(1 << 16, nvcompBatchedLZ4Opts_t{NVCOMP_TYPE_CHAR}, stream, 0);
  const CompressionConfig comp_config = compressor.configure_compression(n);
  size_t lcompbuf = comp_config.max_compressed_buffer_size;

  // The compressed output buffer is padded to the next multiple of 4KB
  // for best I/O performance. Unaligned I/Os go through an extra
  // memory copy (GDS's internal aligned registered buffer)
  lcompbuf = ((lcompbuf - 1) / 4096 + 1) * 4096;
  uint8_t* d_compressed;
  CUDA_CHECK(cudaMalloc(&d_compressed, lcompbuf));

  nvtxRangePop();
  nvtxRangePushA("GDS setup");

  // Initialize the cufile driver
  CUfileError_t status = cuFileDriverOpen();
  if (status.err != CU_FILE_SUCCESS) {
    printf("Error: cuFileDriverOpen failed (%d)\n", status.err);
    return -1;
  }

  // Register the file with GDS
  CUfileDescr_t cf_descr;
  memset(&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUfileHandle_t cf_handle;
  status = cuFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != CU_FILE_SUCCESS) {
    printf("Error: cuFileHandleRegister failed (%d)\n", status.err);
    return -1;
  }

  // Buffer registration is not mandatory but recommended for best performance.
  // I/Os from/to unregistered buffers will go through GDS's internal registered
  // buffer (extra copy).
  // Let's ignore if it fails (e.g. not enough BAR memory on this GPU)
  bool registered = true;
  status = cuFileBufRegister(d_compressed, lcompbuf, 0);
  if (status.err != CU_FILE_SUCCESS) {
    printf("Warning: GDS buffer registration failed\n");
    registered = false;
  }

  nvtxRangePop();
  nvtxRangePushA("Compression");

  // The compressed size must be device-accessible, using pinned memory.

  // Compress the data (asynchronous)
  compressor.compress(d_input, d_compressed, comp_config);
  const size_t compressed_size = compressor.get_compressed_output_size(d_compressed);

  // Align the compressed size to the next multiple of 4KB.
  size_t aligned_compressed_size = ((compressed_size - 1) / 4096 + 1) * 4096;
  printf(
      "Data compressed from %lu Bytes to %lu Bytes, aligned to %lu Bytes\n",
      n,
      compressed_size,
      aligned_compressed_size);

  nvtxRangePop();
  nvtxRangePushA("GDS Write");

  // Write the data (padded to next 4KB), directly from the device, with GDS
  ssize_t nb;
  if ((nb = cuFileWrite(cf_handle, d_compressed, aligned_compressed_size, 0, 0)) != aligned_compressed_size) {
    printf("Error, write returned %ld instead of %lu \n", nb, aligned_compressed_size);
    return -1;
  } else
    printf("Wrote %ld bytes to file %s using GDS\n", nb, filename);

  nvtxRangePop();
  nvtxRangePushA("Cleaning up compressor");

  // Erase the content of the compressed buffer
  cudaMemsetAsync(d_compressed, 0xff, compressed_size, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  nvtxRangePop();
  nvtxRangePushA("GDS Read");

  // Read the compressed data from the GDS file into the device buffer
  if ((nb = cuFileRead(cf_handle, d_compressed, aligned_compressed_size, 0, 0)) != aligned_compressed_size) {
    nvtxRangePop();
    printf("Error, GDS read returned %ld instead of %lu \n", nb, aligned_compressed_size);
    return -1;
  } else {
    nvtxRangePop();
    printf("Read %ld bytes from file %s using GDS\n", nb, filename);
  }

  nvtxRangePushA("Decompressor setup");

  cudaStreamSynchronize(stream);
  // Decompressor, configured with the compressed data
  const DecompressionConfig decomp_config
      = compressor.configure_decompression(comp_config);
  size_t ldecomp = decomp_config.decomp_data_size;
  if (ldecomp != n) {
    printf("Error: Uncompressed size does not match the original size\n");
    return -1;
  }

  // Device-accessible flag to compare the data
  int* dh_invalid;
  CUDA_CHECK(cudaMallocHost(&dh_invalid, sizeof(int)));
  *dh_invalid = 0;

  nvtxRangePop();
  nvtxRangePushA("Decompression and comparison");
  printf("Decompressing\n");

  // Decompress the data (asynchronous)
  compressor.decompress(d_output, d_compressed, decomp_config);
  // Compare the uncompressed data with the original, in the same stream
  compare<<<2 * smcount, 1024, 0, stream>>>(d_input, d_output, dh_invalid, n);

  // Sync the stream before we check the result
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (*dh_invalid)
    printf("FAILED: Uncompressed data does not match the original\n");
  else
    printf("PASSED: Uncompressed data is identical to the input\n");

  nvtxRangePop();
  nvtxRangePushA("Final cleanup");

  // Cleanup
  if (registered) {
    status = cuFileBufDeregister(d_compressed);
    if (status.err != CU_FILE_SUCCESS) {
      printf("Error: cuFileBufDeregister failed(%d)\n", status.err);
      return -1;
    }
  }
  close(fd);
  status = cuFileDriverClose();
  if (status.err != CU_FILE_SUCCESS) {
    printf("Error: cuFileDriverClose failed(%d)\n", status.err);
    return -1;
  }
  CUDA_CHECK(cudaFreeHost(dh_invalid));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_compressed));

  printf("All done, exiting...\n");
  nvtxRangePop();

  return 0;
}