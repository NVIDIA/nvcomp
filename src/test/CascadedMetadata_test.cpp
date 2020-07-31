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

#include "../../tests/catch.hpp"
#include "CascadedMetadata.h"

#include <cstdlib>

using namespace nvcomp;

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("IsSavedRLETest", "[small]")
{
  using T = int;

  const int numElemens = 150;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  nvcompCascadedFormatOpts opts;
  opts.num_RLEs = 1;
  opts.num_deltas = 0;
  opts.use_bp = 0;

  CascadedMetadata metadata(opts, type, sizeof(T) * numElemens, 12);

  // we should have two saved values
  int numSaved = 0;
  for (size_t i = 0; i < metadata.getNumInputs(); ++i) {
    if (metadata.isSaved(i)) {
      ++numSaved;
    }
  }
  REQUIRE(numSaved == 2);
}

TEST_CASE("IsSaved2RLEDeltaBPTest", "[small]")
{
  using T = int;

  const int numElemens = 150;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  nvcompCascadedFormatOpts opts;
  opts.num_RLEs = 2;
  opts.num_deltas = 1;
  opts.use_bp = 1;

  CascadedMetadata metadata(opts, type, sizeof(T) * numElemens, 12);

  // we should have three saved values
  int numSaved = 0;
  for (size_t i = 0; i < metadata.getNumInputs(); ++i) {
    if (metadata.isSaved(i)) {
      ++numSaved;
    }
  }
  REQUIRE(numSaved == 3);
}

TEST_CASE("IsSavedDeltaTest", "[small]")
{
  using T = int;

  const int numElemens = 150;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  nvcompCascadedFormatOpts opts;
  opts.num_RLEs = 0;
  opts.num_deltas = 1;
  opts.use_bp = 0;

  CascadedMetadata metadata(opts, type, sizeof(T) * numElemens, 12);

  // we should have one saved values
  int numSaved = 0;
  for (size_t i = 0; i < metadata.getNumInputs(); ++i) {
    if (metadata.isSaved(i)) {
      ++numSaved;
    }
  }
  REQUIRE(numSaved == 1);
}

TEST_CASE("IsSavedBPTest", "[small]")
{
  using T = int;

  const int numElemens = 150;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  nvcompCascadedFormatOpts opts;
  opts.num_RLEs = 0;
  opts.num_deltas = 0;
  opts.use_bp = 1;

  CascadedMetadata metadata(opts, type, sizeof(T) * numElemens, 12);

  // we should have one saved values
  int numSaved = 0;
  for (size_t i = 0; i < metadata.getNumInputs(); ++i) {
    if (metadata.isSaved(i)) {
      ++numSaved;
    }
  }
  REQUIRE(numSaved == 1);
}

TEST_CASE("GetType2RLEDeltaTest", "[small]")
{
  using T = int;

  const int numElemens = 150;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  nvcompCascadedFormatOpts opts;
  opts.num_RLEs = 2;
  opts.num_deltas = 1;
  opts.use_bp = 0;

  CascadedMetadata metadata(opts, type, sizeof(T) * numElemens, 12);

  // we should have three saved values
  int numByte = 0;
  int numInt = 0;
  for (size_t i = 0; i < metadata.getNumInputs(); ++i) {
    if (metadata.isSaved(i)) {
      if (metadata.getDataType(i) == type) {
        ++numInt;
      } else {
        REQUIRE(metadata.getDataType(i) == NVCOMP_TYPE_UCHAR);
        ++numByte;
      }
    }
  }
  // values
  REQUIRE(numInt == 1);
  // runs
  REQUIRE(numByte == 2);
}

TEST_CASE("GetType2RLEDeltaBPTest", "[small]")
{
  using T = int;

  const int numElemens = 150;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  nvcompCascadedFormatOpts opts;
  opts.num_RLEs = 2;
  opts.num_deltas = 1;
  opts.use_bp = 1;

  CascadedMetadata metadata(opts, type, sizeof(T) * numElemens, 12);

  // we should have three saved values
  for (size_t i = 0; i < metadata.getNumInputs(); ++i) {
    if (metadata.isSaved(i)) {
      // all values should be bits
      REQUIRE(metadata.getDataType(i) == NVCOMP_TYPE_BITS);
    }
  }
}
