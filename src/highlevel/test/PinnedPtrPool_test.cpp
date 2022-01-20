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

#define CATCH_CONFIG_MAIN

#include <vector>
#include "cuda_runtime.h"

#include "tests/catch.hpp"
#include "common.h"

#include "highlevel/PinnedPtrs.hpp"

using namespace nvcomp;
using namespace std;

void test_pinned_ptrs() {
  PinnedPtrPool<int> pool{};
  typedef PinnedPtrWrapper<int> PinnedPtr;
  REQUIRE(pool.get_alloced_size() == PINNED_POOL_PREALLOC_SIZE);
  REQUIRE(pool.get_current_pool_size() == PINNED_POOL_PREALLOC_SIZE);

  vector<PinnedPtr> pinned_ptrs;
  for (int i = 1; i <= PINNED_POOL_PREALLOC_SIZE; ++i)
  {
    pinned_ptrs.push_back(PinnedPtr{pool});
    
    REQUIRE(pool.get_current_pool_size() == PINNED_POOL_PREALLOC_SIZE - i);
    REQUIRE(pool.get_alloced_size() == PINNED_POOL_PREALLOC_SIZE);
    
    *pinned_ptrs.back() = i;
  }

  REQUIRE(pool.get_alloced_size() == PINNED_POOL_PREALLOC_SIZE);
  REQUIRE(pool.get_current_pool_size() == 0);

  pinned_ptrs.pop_back(); // return one to the pool
  REQUIRE(pool.get_alloced_size() == PINNED_POOL_PREALLOC_SIZE);
  REQUIRE(pool.get_current_pool_size() == 1);

  for (int i = 0; i < 2; ++i) {
    pinned_ptrs.push_back(PinnedPtr{pool});
  }
  REQUIRE(pool.get_current_pool_size() == PINNED_POOL_REALLOC_SIZE - 1);
  REQUIRE(pool.get_alloced_size() == PINNED_POOL_REALLOC_SIZE + PINNED_POOL_PREALLOC_SIZE);

  pinned_ptrs.clear();
  REQUIRE(pool.get_alloced_size() == PINNED_POOL_REALLOC_SIZE + PINNED_POOL_PREALLOC_SIZE);
  REQUIRE(pool.get_current_pool_size() == PINNED_POOL_REALLOC_SIZE + PINNED_POOL_PREALLOC_SIZE);

}

TEST_CASE("test_pinned_ptrs")
{
  test_pinned_ptrs();
}
