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

#include "../src/common.h"
#include "cascaded.h"
#include "nvcomp.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw;                                                                   \
    }                                                                          \
  } while (0);

namespace nvcomp
{

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
template <>
inline nvcompType_t getnvcompType<float>()
{
  return NVCOMP_TYPE_INT;
}

inline bool startsWith(const std::string input, const std::string subStr)
{
  return input.substr(0, subStr.length()) == subStr;
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

void benchmark_assert(const bool pass, const std::string& msg)
{
  if (!pass) {
    throw std::runtime_error("ERROR: " + msg);
  }
}

// Load dataset from binary file into an array of type T
template <typename T>
std::vector<T> load_dataset_from_binary(char* fname, size_t* input_elts)
{

  FILE* fileptr;

  fileptr = fopen(fname, "rb");

  if (fileptr == NULL) {
    printf("Binary input file not found.\n");
    exit(1);
  }

  // find length of file
  fseek(fileptr, 0, SEEK_END);
  size_t filelen = ftell(fileptr);
  rewind(fileptr);

  // If input_elts is already set, use it, otherwise load the whole file
  if (*input_elts == 0 || filelen / sizeof(T) < *input_elts) {
    *input_elts = filelen / sizeof(T);
  }

  const size_t numElements = *input_elts;

  std::vector<T> buffer(numElements);

  // Read binary file in to buffer
  const size_t numRead = fread(buffer.data(), sizeof(T), numElements, fileptr);
  if (numRead != numElements) {
    throw std::runtime_error(
        "Failed to read file: " + std::string(fname) + " read "
        + std::to_string(numRead) + "/"
        + std::to_string(*input_elts * sizeof(T)) + " elements.");
  }

  fclose(fileptr);
  return buffer;
}

// Load dataset from binary file into an array of type T
template <typename T>
std::vector<T> load_dataset_from_txt(char* fname, size_t* input_elts)
{

  std::vector<T> buffer;
  FILE* fileptr;

  fileptr = fopen(fname, "rb");

  if (fileptr == NULL) {
    printf("Text input file not found.\n");
    exit(1);
  }

  size_t i = 0;
  constexpr size_t MAX_LINE_LEN = 100;
  char line[MAX_LINE_LEN];
  while (fgets(line, MAX_LINE_LEN, fileptr) && i < *input_elts) {
    //    std::stringstream row(line);
    buffer.push_back((T)std::stof(line));
    i++;
  }

  fclose(fileptr);

  return buffer;
}

// Compress a single chunk
template <typename T>
static void compress_chunk(
    const void* const d_in_data,
    const size_t chunk_size,
    const nvcompType_t type,
    const nvcompCascadedFormatOpts* const comp_opts,
    void* const d_comp_temp,
    const size_t comp_temp_bytes,
    void* const d_comp_out,
    size_t* const comp_out_bytes,
    cudaStream_t stream)
{

  nvcompError_t status = nvcompCascadedCompressGetOutputSize(
      d_in_data,
      chunk_size,
      type,
      comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      comp_out_bytes,
      false);

  benchmark_assert(
      status == nvcompSuccess,
      "nvcompCascadedCompressGetMetadata not successful, chunk");

  status = nvcompCascadedCompressAsync(
      d_in_data,
      chunk_size,
      type,
      comp_opts,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes,
      stream);

  benchmark_assert(
      status == nvcompSuccess,
      "nvcompCascadedCompressAsync not successfully launched");
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace nvcomp
