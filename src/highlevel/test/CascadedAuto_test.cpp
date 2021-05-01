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

#define CATCH_CONFIG_MAIN

#include "nvcomp.h"
#include "nvcomp.hpp"
#include "nvcomp/cascaded.h"
#include "nvcomp/cascaded.hpp"

#include "../../../tests/catch.hpp"
#include "../CascadedCommon.h"
#include "../CascadedCompressionGPU.h"
#include "common.h"
#include "type_macros.h"

#include "cuda_runtime.h"

#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                     \
  {                                                                            \
    cudaError_t cudaStatus = call;                                             \
    if (cudaSuccess != cudaStatus) {                                           \
      fprintf(                                                                 \
          stderr,                                                              \
          "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s "   \
          "(%d).\n",                                                           \
          #call,                                                               \
          __LINE__,                                                            \
          __FILE__,                                                            \
          cudaGetErrorString(cudaStatus),                                      \
          cudaStatus);                                                         \
      abort();                                                                 \
    }                                                                          \
  }
#endif

using namespace nvcomp;
using namespace std;

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("AutoTempSize_OutputSize_C", "[small]")
{

  size_t const n = 1000000;
  using T = int32_t;

  T* d_input;
  const size_t numBytes = n * sizeof(T);

  CUDA_RT_CALL(cudaMalloc(&d_input, numBytes));

  size_t metadata_bytes = 0;
  size_t temp_bytes = 0;
  size_t compressed_bytes = 0;

  nvcompError_t err = nvcompCascadedCompressConfigure(
      NULL,
      TypeOf<T>(),
      numBytes,
      &metadata_bytes,
      &temp_bytes,
      &compressed_bytes);

  REQUIRE(err == nvcompSuccess);
  REQUIRE(temp_bytes == 12251072);
  REQUIRE(compressed_bytes == 12000256);

}

// TODO - Add more unit tests of auto-run API calls
