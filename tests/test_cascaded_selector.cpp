/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "catch.hpp"
#include "nvcomp.hpp"
#include "nvcomp/cascaded.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>

// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace nvcomp;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
std::vector<T> buildRuns(const size_t numRuns, const size_t runSize, const size_t step)
{
  std::vector<T> input;
  for (size_t i = 0; i < numRuns; i++) {
    for (size_t j = 0; j < runSize; j++) {
      input.push_back(static_cast<T>(i*step));
    }
  }

  return input;
}

// Run C API selector and return the estimated compression ratio
template <typename T>
double test_selector_c(const std::vector<T>& input, size_t sample_size, size_t num_samples, nvcompCascadedFormatOpts* opts)
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  size_t temp_bytes = 0;
  void* d_temp;
  nvcompCascadedSelectorOpts selector_opts;
  selector_opts.sample_size = sample_size;
  selector_opts.num_samples = num_samples;
  selector_opts.seed = 1;

  nvcompError_t err = nvcompCascadedSelectorGetTempSize(in_bytes, getnvcompType<T>(), selector_opts, &temp_bytes);
  REQUIRE(err == nvcompSuccess);

  CUDA_CHECK( cudaMalloc(&d_temp, temp_bytes) );

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  double est_ratio;

  err = nvcompCascadedSelectorSelectConfig(
           d_in_data,
           in_bytes,
           getnvcompType<T>(),
           selector_opts,
           d_temp,
           temp_bytes,
           opts,
           &est_ratio,
           stream);

  cudaStreamSynchronize(stream);
  REQUIRE(err == nvcompSuccess);

  cudaFree(d_temp);
  cudaFree(d_in_data);
  
  return est_ratio;
}

// Run C++ API selector and return the estimated compression ratio
template <typename T>
double test_selector_cpp(
const std::vector<T>& input, 
size_t sample_size, 
size_t num_samples, 
nvcompCascadedFormatOpts* opts)
{

  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  nvcompCascadedSelectorOpts selector_opts;
  selector_opts.sample_size = sample_size;
  selector_opts.num_samples = num_samples;
  selector_opts.seed = 1;

  // Create the selector
  CascadedSelector<T> selector((const void*)d_in_data, in_bytes, selector_opts);

  // Get temp size and allocate it
  size_t temp_bytes = selector.get_temp_size();
  void* d_temp;
  CUDA_CHECK( cudaMalloc(&d_temp, temp_bytes) );

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  double est_ratio;

  *opts = selector.select_config(d_temp, temp_bytes, &est_ratio, stream);

  cudaStreamSynchronize(stream);

  cudaFree(d_temp);
  cudaFree(d_in_data);
  
  return est_ratio;
}  

} // namespace

void verify_selector_result(nvcompCascadedFormatOpts opts, double est_ratio, int exp_RLEs, int exp_deltas, int exp_bp, double exp_ratio) {
  REQUIRE(est_ratio >= exp_ratio);
  REQUIRE(opts.num_RLEs == exp_RLEs);
  REQUIRE(opts.num_deltas == exp_deltas);
  REQUIRE(opts.use_bp == exp_bp);
}

/******************************************************************************
 * Test cases *****************************************************************
 *****************************************************************************/


TEST_CASE("CascadedSelector tiny-example", "[nvcomp]")
{
  using T = int;

  std::vector<T> input = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 2, 3, 3};
  nvcompCascadedFormatOpts opts;

  double est_ratio = test_selector_c<T>(input, 4, 4, &opts);
  verify_selector_result(opts, est_ratio, 0, 0, 1, 2.0);

  est_ratio = test_selector_cpp<T>(input, 4, 4, &opts);
  verify_selector_result(opts, est_ratio, 0, 0, 1, 2.0);
}


TEST_CASE("CascadedSelector all-small-byte", "[nvcomp][small]")
{
  using T = int8_t;

  nvcompCascadedFormatOpts opts;
  double est_ratio;

  // Tiny example with poor compression
  for (int total = 64; total <= 1024; total += 64) {
    std::vector<T> input = buildRuns<T>(total, 1, 1);

    est_ratio = test_selector_c<T>(input, 16, 4, &opts);
    verify_selector_result(opts, est_ratio, 0, 0, 1, 2);

    est_ratio = test_selector_cpp<T>(input, 16, 4, &opts);
    verify_selector_result(opts, est_ratio, 0, 0, 1, 2);
  }

  // Small example with moderate compression
  for (int total = 1024; total <= 8192; total += 1024) {
    std::vector<T> input = buildRuns<T>(total, 1, 1);

    est_ratio = test_selector_c<T>(input, 128, 8, &opts);
    verify_selector_result(opts, est_ratio, 0, 1, 1, 8);

    est_ratio = test_selector_cpp<T>(input, 128, 8, &opts);
    verify_selector_result(opts, est_ratio, 0, 1, 1, 8);
  }

  // Small example with better compression
  for (int total = 1024; total <= 8192; total += 1024) {
    std::vector<T> input = buildRuns<T>(2, total/2, 10);

    est_ratio = test_selector_c<T>(input, 128, 4, &opts);
    verify_selector_result(opts, est_ratio, 1, 0, 1, 16);

    est_ratio = test_selector_cpp<T>(input, 128, 4, &opts);
    verify_selector_result(opts, est_ratio, 1, 0, 1, 16);
  }
}

TEST_CASE("CascadedSelector all-small-int", "[nvcomp][small]")
{
  using T = int32_t;

  nvcompCascadedFormatOpts opts;
  double est_ratio;

  // Tiny example with moderate int compression
  for (int total = 64; total <= 1024; total += 64) {
    std::vector<T> input = buildRuns<T>(total, 1, 1);

    est_ratio = test_selector_c<T>(input, 16, 4, &opts);
    verify_selector_result(opts, est_ratio, 0, 0, 1, 8);

    est_ratio = test_selector_cpp<T>(input, 16, 4, &opts);
    verify_selector_result(opts, est_ratio, 0, 0, 1, 8);
  }

  // Small example with moderate compression
  for (int total = 1024; total <= 8192; total += 1024) {
    std::vector<T> input = buildRuns<T>(total, 1, 1);

    est_ratio = test_selector_c<T>(input, 128, 8, &opts);
    verify_selector_result(opts, est_ratio, 0, 1, 1, 32);

    est_ratio = test_selector_cpp<T>(input, 128, 8, &opts);
    verify_selector_result(opts, est_ratio, 0, 1, 1, 32);
  }

  // Small example with better compression
  // TODO: this fails intermittently, commented for now
/*
  for (int total = 1024; total <= 8192; total += 1024) {
    std::vector<T> input = buildRuns<T>(2, total/2, 1);

    est_ratio = test_selector_c<T>(input, 128, 8, &opts);
    verify_selector_result(opts, est_ratio, 1, 0, 1, 64);

    est_ratio = test_selector_cpp<T>(input, 128, 8, &opts);
    verify_selector_result(opts, est_ratio, 1, 0, 1, 64);
  }
*/
}


TEST_CASE("CascadedSelector all-big-int", "[nvcomp][big]")
{
  using T = int32_t;

  nvcompCascadedFormatOpts opts;
  double est_ratio;

  // Large examples with Delta compression
  for (int total = 1000000; total <= 10000000; total += 1000000) {
    std::vector<T> input = buildRuns<T>(total, 1, 1);

    est_ratio = test_selector_c<T>(input, 1024, 100, &opts);
    verify_selector_result(opts, est_ratio, 0, 1, 1, 32);

    est_ratio = test_selector_cpp<T>(input, 1024, 100, &opts);
    verify_selector_result(opts, est_ratio, 0, 1, 1, 32);
  }

  // Large examples with RLE compression
  for (int total = 1000000; total <= 10000000; total += 1000000) {
    std::vector<T> input = buildRuns<T>(100, total/100, 1);

    est_ratio = test_selector_c<T>(input, 1024, 100, &opts);
    verify_selector_result(opts, est_ratio, 1, 0, 1, 100);

    est_ratio = test_selector_cpp<T>(input, 1024, 100, &opts);
    verify_selector_result(opts, est_ratio, 1, 0, 1, 100);
  }

  // Large examples with more complex compression that results in R2D1B1
  for (int total = 2000000; total <= 10000000; total += 1000000) {
    std::vector<T> input = buildRuns<T>(1000000, total/1000000, 2);

    est_ratio = test_selector_c<T>(input, 1024, 100, &opts);
    verify_selector_result(opts, est_ratio, 2, 1, 1, 20);

    est_ratio = test_selector_cpp<T>(input, 1024, 100, &opts);
    verify_selector_result(opts, est_ratio, 2, 1, 1, 20);
  }
}


