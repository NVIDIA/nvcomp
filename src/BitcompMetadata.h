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

#ifndef NVCOMP_BITCOMPMETADATA_H
#define NVCOMP_BITCOMPMETADATA_H

#ifdef ENABLE_BITCOMP
#include "bitcomp.h"
#else
typedef int bitcompHandle_t;
#endif

#include "Metadata.h"

namespace nvcomp
{

class BitcompMetadata : public Metadata
{
public:
  constexpr static int COMPRESSION_ID = 0x5000;

  /**
   * @brief Create metadata object from compressed data either on CPU or GPU.
   *
   * @param memPtr The pointer (host or device memory) containing the compressed
   * data.
   * @param compressedBytes The total size of the data in memPtr
   */
  BitcompMetadata(const void* const memPtr, size_t compressedBytes);

#ifdef ENABLE_BITCOMP
  ~BitcompMetadata()
  {
    bitcompDestroyPlan(plan);
  }
  bitcompHandle_t getBitcompHandle()
  {
    return plan;
  }
#endif

// disable copying
  BitcompMetadata(const BitcompMetadata&) = delete;
  BitcompMetadata& operator=(const BitcompMetadata&) = delete;

private:
  /**
   * @brief A bitcomp handle holds all the info about the compressed data except
   * the compressed size and can be populated from compressed data.
   */
  bitcompHandle_t plan;
};

} // namespace nvcomp

#endif