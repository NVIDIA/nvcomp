/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef VERBOSE
#define VERBOSE 0
#endif

#include "test_common.h"

#include <thread>
#include <vector>

template <typename T>
void test_random(
    int max_val,
    int max_run,
    size_t chunk_size,
    int numRLEs,
    int numDeltas,
    int bitPacking)
{
  // generate random data
  std::vector<T> data;
  int seed = (max_val ^ max_run ^ chunk_size);
  random_runs(data, (T)max_val, (T)max_run, seed);

  test<T>(data, chunk_size, numRLEs, numDeltas, bitPacking);
}

template <typename T>
void test_random_mt(
    int num_threads,
    int max_val,
    int max_run,
    size_t chunk_size,
    int numRLEs,
    int numDeltas,
    int bitPacking)
{
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++)
    threads.push_back(std::thread(
        &test_random<T>,
        max_val,
        max_run,
        chunk_size,
        numRLEs,
        numDeltas,
        bitPacking));

  for (int i = 0; i < num_threads; i++)
    threads[i].join();
}

// int: single-threaded
TEST_CASE("small-R-int", "[small]")
{
  test_random<int>(10, 10, 10000, 1, 0, 0);
}
TEST_CASE("small-D-int", "[small]")
{
  test_random<int>(10, 10, 10000, 0, 1, 0);
}
TEST_CASE("small-b-int", "[small][bp]")
{
  test_random<int>(10, 10, 10000, 0, 0, 1);
}
TEST_CASE("small-RD-int", "[small]")
{
  test_random<int>(10, 10, 10000, 1, 1, 0);
}
TEST_CASE("small-Db-int", "[small][bp]")
{
  test_random<int>(10, 10, 10000, 0, 1, 1);
}
TEST_CASE("small-Rb-int", "[small][bp]")
{
  test_random<int>(10, 10, 10000, 1, 0, 1);
}
TEST_CASE("small-RR-int", "[small]")
{
  test_random<int>(10, 10, 10000, 2, 0, 0);
}
TEST_CASE("small-DD-int", "[small]")
{
  test_random<int>(10, 10, 10000, 0, 2, 0);
}
TEST_CASE("small-RDR-int", "[small]")
{
  test_random<int>(10, 10, 10000, 2, 1, 0);
}
TEST_CASE("small-RDb-int", "[small][bp]")
{
  test_random<int>(10, 10, 10000, 1, 1, 1);
}
TEST_CASE("small-RRb-int", "[small][bp]")
{
  test_random<int>(10, 10, 10000, 2, 0, 1);
}
TEST_CASE("small-DDb-int", "[small][bp]")
{
  test_random<int>(10, 10, 10000, 0, 2, 1);
}
TEST_CASE("small-RDRb-int", "[small][bp]")
{
  test_random<int>(10, 10, 10000, 2, 1, 1);
}
TEST_CASE("small-RDDb-int", "[small][bp]")
{
  test_random<int>(10, 10, 10000, 1, 2, 1);
}
TEST_CASE("small-RDRDb-int", "[small][bp]")
{
  test_random<int>(10, 10, 10000, 2, 2, 1);
}
TEST_CASE("small-RDRRb-int", "[small][bp]")
{
  test_random<int>(10, 10, 10000, 3, 1, 1);
}

TEST_CASE("large-b-int", "[large][bp]")
{
  test_random<int>(10000, 1000, 10000000, 0, 0, 1);
}
TEST_CASE("large-R-int", "[large]")
{
  test_random<int>(10000, 1000, 10000000, 1, 0, 0);
}
TEST_CASE("large-RD-int", "[large]")
{
  test_random<int>(10000, 1000, 10000000, 1, 1, 0);
}
TEST_CASE("large-RDR-int", "[large]")
{
  test_random<int>(10000, 1000, 10000000, 2, 1, 0);
}
TEST_CASE("large-Rb-int", "[large][bp]")
{
  test_random<int>(10000, 1000, 10000000, 1, 0, 1);
}
TEST_CASE("large-Db-int", "[large][bp]")
{
  test_random<int>(10000, 1000, 10000000, 0, 1, 1);
}
TEST_CASE("large-RDb-int", "[large][bp]")
{
  test_random<int>(10000, 1000, 10000000, 1, 1, 1);
}
TEST_CASE("large-RRb-int", "[large][bp]")
{
  test_random<int>(10000, 1000, 10000000, 2, 0, 1);
}
TEST_CASE("large-DDb-int", "[large][bp]")
{
  test_random<int>(10000, 1000, 10000000, 0, 2, 1);
}
TEST_CASE("large-RDRb-int", "[large][bp]")
{
  test_random<int>(10000, 1000, 10000000, 2, 1, 1);
}
TEST_CASE("large-RDDb-int", "[large][bp]")
{
  test_random<int>(10000, 1000, 10000000, 1, 2, 1);
}

// int: multi-threaded
TEST_CASE("large-R-int-t10", "[large][mt]")
{
  test_random_mt<int>(10, 10000, 1000, 10000000, 1, 0, 0);
}
TEST_CASE("large-RD-int-t10", "[large][mt]")
{
  test_random_mt<int>(10, 10000, 1000, 10000000, 1, 1, 0);
}
TEST_CASE("large-RDR-int-t10", "[large][mt]")
{
  test_random_mt<int>(10, 10000, 1000, 10000000, 2, 1, 0);
}
TEST_CASE("large-Rb-int-t10", "[large][bp][mt]")
{
  test_random_mt<int>(10, 10000, 1000, 10000000, 1, 0, 1);
}
TEST_CASE("large-RDb-int-t10", "[large][bp][mt]")
{
  test_random_mt<int>(10, 10000, 1000, 10000000, 1, 1, 1);
}
TEST_CASE("large-RDRb-int-t10", "[large][bp][mt]")
{
  test_random_mt<int>(10, 10000, 1000, 10000000, 2, 1, 1);
}

// long long
TEST_CASE("small-R-ll", "[small]")
{
  test_random<int64_t>(10, 10, 10000, 1, 0, 0);
}
TEST_CASE("small-RD-ll", "[small]")
{
  test_random<int64_t>(10, 10, 10000, 1, 1, 0);
}
TEST_CASE("small-RDR-ll", "[small]")
{
  test_random<int64_t>(10, 10, 10000, 2, 1, 0);
}
TEST_CASE("small-Rb-ll", "[small][bp]")
{
  test_random<int64_t>(10, 10, 10000, 1, 0, 1);
}
TEST_CASE("small-RDb-ll", "[small][bp]")
{
  test_random<int64_t>(10, 10, 10000, 1, 1, 1);
}
TEST_CASE("small-RDRb-ll", "[small][bp]")
{
  test_random<int64_t>(10, 10, 10000, 2, 1, 1);
}

TEST_CASE("large-R-ll", "[large]")
{
  test_random<int64_t>(10000, 1000, 10000000, 1, 0, 0);
}
TEST_CASE("large-RD-ll", "[large]")
{
  test_random<int64_t>(10000, 1000, 10000000, 1, 1, 0);
}
TEST_CASE("large-RDR-ll", "[large]")
{
  test_random<int64_t>(10000, 1000, 10000000, 2, 1, 0);
}
TEST_CASE("large-Rb-ll", "[large][bp]")
{
  test_random<int64_t>(10000, 1000, 10000000, 1, 0, 1);
}
TEST_CASE("large-RDb-ll", "[large][bp]")
{
  test_random<int64_t>(10000, 1000, 10000000, 1, 1, 1);
}
TEST_CASE("large-RDRb-ll", "[large][bp]")
{
  test_random<int64_t>(10000, 1000, 10000000, 2, 1, 1);
}

TEST_CASE("large-RLE-expand", "[large][bp]")
{
  const int n = 100 * 1000 * 1000;
  const int val = 74252534;
  std::vector<int> data(n, val);
  test(data, n, 1, 0, 1);
}
