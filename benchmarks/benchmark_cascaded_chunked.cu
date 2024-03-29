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

#include "benchmark_template_chunked.cuh"
#include "nvcomp/cascaded.h"

#include <iostream>
#include <vector>

static nvcompBatchedCascadedOpts_t nvcompBatchedCascadedTestOpts
    = {4096, NVCOMP_TYPE_UINT, 2, 1, 1};

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
  if (arg == "--type" || arg == "-t") {
    const char* const typeArg = *additionalArgs;
    additionalArgsUsed = 1;
    bool valid;
    nvcompBatchedCascadedTestOpts.type = string_to_data_type(typeArg, valid);
    return valid;
  }
  if (arg == "--num_rles" || arg == "-r") {
    int n = atoi(*additionalArgs);
    additionalArgsUsed = 1;
    if (n < 0) {
      std::cerr << "ERROR: num_rles can't be negative, but it is " << n
                << std::endl;
      return false;
    }
    nvcompBatchedCascadedTestOpts.num_RLEs = n;
    return true;
  }
  if (arg == "--num_deltas" || arg == "-d") {
    int n = atoi(*additionalArgs);
    additionalArgsUsed = 1;
    if (n < 0) {
      std::cerr << "ERROR: num_deltas can't be negative, but it is " << n
                << std::endl;
      return false;
    }
    nvcompBatchedCascadedTestOpts.num_deltas = n;
    return true;
  }
  if (arg == "--num_bps" || arg == "-b") {
    int n = atoi(*additionalArgs);
    additionalArgsUsed = 1;
    if (n < 0 || n > 1) {
      std::cerr << "ERROR: num_bps can only be 0 or 1, but it is " << n
                << std::endl;
      return false;
    }
    nvcompBatchedCascadedTestOpts.use_bp = n;
    return true;
  }
  return false;
}

static bool isCascadedInputValid(const std::vector<std::vector<char>>& data)
{
  // Find the type size, to check that all chunk sizes are a multiple of it.
  size_t typeSize = 1;
  auto type = nvcompBatchedCascadedTestOpts.type;
  switch (type) {
  case NVCOMP_TYPE_CHAR:
  case NVCOMP_TYPE_UCHAR:
    // Type size is 1 byte, so chunk sizes are always a multiple of it.
    return true;
  case NVCOMP_TYPE_SHORT:
  case NVCOMP_TYPE_USHORT:
    typeSize = sizeof(uint16_t);
    break;
  case NVCOMP_TYPE_INT:
  case NVCOMP_TYPE_UINT:
    typeSize = sizeof(uint32_t);
    break;
  case NVCOMP_TYPE_LONGLONG:
  case NVCOMP_TYPE_ULONGLONG:
    typeSize = sizeof(uint64_t);
    break;
  default:
    std::cerr << "ERROR: Cascaded data type must be 0-7 (CHAR, UCHAR, SHORT, "
                 "USHORT, INT, UINT, LONGLONG, or ULONGLONG), "
                 "but it is "
              << int(type) << std::endl;
    return false;
  }

  for (const auto& chunk : data) {
    if ((chunk.size() % typeSize) != 0) {
      std::cerr << "ERROR: Input data must have a length and chunk size that "
                   "are a multiple of "
                << typeSize << ", the size of the specified data type."
                << std::endl;
      return false;
    }
  }
  return true;
}

void run_benchmark(
    const std::vector<std::vector<char>>& data,
    const bool warmup,
    const size_t count,
    const bool csv_output,
    const bool tab_separator,
    const size_t duplicate_count,
    const size_t num_files)
{
  run_benchmark_template(
      nvcompBatchedCascadedCompressGetTempSize,
      nvcompBatchedCascadedCompressGetMaxOutputChunkSize,
      nvcompBatchedCascadedCompressAsync,
      nvcompBatchedCascadedDecompressGetTempSize,
      nvcompBatchedCascadedDecompressAsync,
      isCascadedInputValid,
      nvcompBatchedCascadedTestOpts,
      data,
      warmup,
      count,
      csv_output,
      tab_separator,
      duplicate_count,
      num_files);
}
