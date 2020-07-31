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

#include "LZ4CompressionKernels.h"
#include "TempSpaceBroker.h"
#include "common.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <cub/cub.cuh>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// align all temp allocations by 512B
#define CUDA_MEM_ALIGN(size) (((size) + 0x1FF) & ~0x1FF)

#include "cuda_runtime.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

using offset_type = uint16_t;
using word_type = uint32_t;
using position_type = size_t;
using double_word_type = uint64_t;

namespace nvcomp {

constexpr const int DECOMP_THREADS = 32;
constexpr const int Y_DIM = 2;
constexpr const position_type BUFFER_SIZE
    = DECOMP_THREADS * sizeof(double_word_type);
constexpr const position_type PREFETCH_DIST = BUFFER_SIZE / 2;

constexpr const position_type HASH_TABLE_SIZE = 1U << 14;
constexpr const offset_type NULL_OFFSET = static_cast<offset_type>(-1);
constexpr const position_type MAX_OFFSET = (1U << 16) - 1;

struct block_stats_st
{
  uint64_t cycles;
  int copy_length_min;
  int copy_length_max;
  int copy_length_sum;
  int copy_length_count;
  int copy_lsic_count;
  int match_length_min;
  int match_length_max;
  int match_length_sum;
  int match_length_count;
  int match_lsic_count;
  int match_overlaps;
  int offset_min;
  int offset_max;
  int offset_sum;
};

/******************************************************************************
 * DEVICE FUNCTIONS AND KERNELS ***********************************************
 *****************************************************************************/

inline __device__ void syncCTA()
{
  if (DECOMP_THREADS > 32) {
    __syncthreads();
  } else {
    __syncwarp();
  }
}

template <typename T>
inline __device__ void writeWord(uint8_t* const address, const T word)
{
#pragma unroll
  for (size_t i = 0; i < sizeof(T); ++i) {
    address[i] = static_cast<uint8_t>((word >> (8 * i)) & 0xff);
  }
}

template <typename T>
inline __device__ T readWord(const uint8_t* const address)
{
  T word = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    word |= address[i] << (8 * i);
  }

  return word;
}
inline __device__ void writeLSIC(uint8_t* const out, position_type number)
{
  size_t i = 0;
  while (number >= 0xff) {
    out[i] = 0xff;
    ++i;
    number -= 0xff;
  }
  out[i] = number;
}

struct token_type
{
  position_type num_literals;
  position_type num_matches;

   __device__ bool hasNumLiteralsOverflow() const
  {
    return num_literals >= 15;
  }

   __device__ bool hasNumMatchesOverflow() const
  {
    return num_matches >= 19;
  }

  __device__ position_type numLiteralsOverflow() const
  {
    if (hasNumLiteralsOverflow()) {
      return num_literals - 15;
    } else {
      return 0;
    }
  }

  __device__ uint8_t numLiteralsForHeader() const
  {
    if (hasNumLiteralsOverflow()) {
      return 15;
    } else {
      return num_literals;
    }
  }

  __device__ position_type numMatchesOverflow() const
  {
    if (hasNumMatchesOverflow()) {
      assert(num_matches >= 19);
      return num_matches - 19;
    } else {
      assert(num_matches < 19);
      return 0;
    }
  }

  __device__ uint8_t numMatchesForHeader() const
  {
    if (hasNumMatchesOverflow()) {
      return 15;
    } else {
      return num_matches - 4;
    }
  }
  __device__ position_type lengthOfLiteralEncoding() const
  {
    if (hasNumLiteralsOverflow()) {
      position_type length = 1;
      position_type num = numLiteralsOverflow();
      while (num >= 0xff) {
        num -= 0xff;
        length += 1;
      }

      return length;
    }
    return 0;
  }

  __device__ position_type lengthOfMatchEncoding() const
  {
    if (hasNumMatchesOverflow()) {
      position_type length = 1;
      position_type num = numMatchesOverflow();
      while (num >= 0xff) {
        num -= 0xff;
        length += 1;
      }

      return length;
    }
    return 0;
  }
};

class BufferControl
{
public:

  __device__ BufferControl(
      uint8_t* const buffer, const uint8_t* const compData, const position_type length) :
      m_offset(0),
      m_length(length),
      m_buffer(buffer),
      m_compData(compData)
  {
    // do nothing
  }

  #ifdef WARP_READ_LSIC
    // this is currently unused as its slower
  inline __device__ position_type queryLSIC(const position_type idx) const
  {
    if (idx + DECOMP_THREADS <= end()) {
      // most likely case
      const uint8_t byte = rawAt(idx)[threadIdx.x];
  
      uint32_t mask = __ballot_sync(0xffffffff, byte != 0xff);
      mask = __brev(mask);
  
      const position_type fullBytes = __clz(mask);
  
      if (fullBytes < DECOMP_THREADS) {
        return fullBytes * 0xff + rawAt(idx)[fullBytes];
      } else {
        return DECOMP_THREADS * 0xff;
      }
    } else {
      uint8_t byte;
      if (idx + threadIdx.x < end()) {
        byte = rawAt(idx)[threadIdx.x];
      } else {
        byte = m_compData[idx + threadIdx.x];
      }
  
      uint32_t mask = __ballot_sync(0xffffffff, byte != 0xff);
      mask = __brev(mask);
  
      const position_type fullBytes = __clz(mask);
  
      if (fullBytes < DECOMP_THREADS) {
        return fullBytes * 0xff + __shfl_sync(0xffffffff, byte, fullBytes);
      } else {
        return DECOMP_THREADS * 0xff;
      }
    }
  }
  #endif
  
  inline __device__ position_type readLSIC(position_type& idx) const
  {
  #ifdef WARP_READ_LSIC
    position_type num = 0;
    while (true) {
      const position_type block = queryLSIC(idx);
      num += block;
  
      if (block < DECOMP_THREADS * 0xff) {
        idx += (block / 0xff) + 1;
        break;
      } else {
        idx += DECOMP_THREADS;
      }
    }
    return num;
  #else
    position_type num = 0;
    uint8_t next = 0xff;
    // read from the buffer
    while (next == 0xff && idx < end()) {
      next = rawAt(idx)[0];
      ++idx;
      num += next;
    }
      // read from global memory
    while (next == 0xff) {
      next = m_compData[idx];
      ++idx;
      num += next;
    }
    return num;
  #endif
  }
  
    inline __device__ const uint8_t* raw() const
    {
      return m_buffer;
    }
  
    inline __device__ const uint8_t* rawAt(const position_type i) const
    {
      return raw() + (i - begin());
    }
    inline __device__ uint8_t operator[](const position_type i) const
    {
      if (i >= m_offset && i - m_offset < BUFFER_SIZE) {
        return m_buffer[i - m_offset];
      } else {
        return m_compData[i];
      }
    }
  
  inline __device__ void loadAt(const position_type offset)
  {
    m_offset = (offset / sizeof(double_word_type)) * sizeof(double_word_type);
  
    if (m_offset + BUFFER_SIZE <= m_length) {
      assert(m_offset % sizeof(double_word_type) == 0);
      assert(BUFFER_SIZE == DECOMP_THREADS * sizeof(double_word_type));
      const double_word_type* const word_data
          = reinterpret_cast<const double_word_type*>(m_compData + m_offset);
      double_word_type* const word_buffer
          = reinterpret_cast<double_word_type*>(m_buffer);
      word_buffer[threadIdx.x] = word_data[threadIdx.x];
    } else {
  #pragma unroll
      for (int i = threadIdx.x; i < BUFFER_SIZE; i += DECOMP_THREADS) {
        if (m_offset + i < m_length) {
          m_buffer[i] = m_compData[m_offset + i];
        }
      }
    }
  
    syncCTA();
  }
  
  inline __device__ position_type begin() const
  {
    return m_offset;
  }
  
  
  inline __device__ position_type end() const
  {
    return m_offset + BUFFER_SIZE;
  }

private:
  position_type m_offset;
  const position_type m_length;
  uint8_t* const m_buffer;
  const uint8_t* const m_compData;
}; //End BufferControl Class


inline __device__ void coopCopyNoOverlap(
    uint8_t* const dest, const uint8_t* const source, const size_t length)
{
  for (size_t i = threadIdx.x; i < length; i += blockDim.x) {
    dest[i] = source[i];
  }
}

inline __device__ void coopCopyRepeat(
    uint8_t* const dest,
    const uint8_t* const source,
    const position_type dist,
    const position_type length)
{
// if there is overlap, it means we repeat, so we just
// need to organize our copy around that
  for (position_type i = threadIdx.x; i < length; i += blockDim.x) {
    dest[i] = source[i % dist];
  }
}

inline __device__ void coopCopyOverlap(
    uint8_t* const dest,
    const uint8_t* const source,
    const position_type dist,
    const position_type length)
{
  if (dist < length) {
    coopCopyRepeat(dest, source, dist, length);
  } else {
    coopCopyNoOverlap(dest, source, length);
  }
}

inline __device__ position_type hash(const word_type key)
{
  // needs to be 12 bits
//  return ((key >> 16) + key) & (HASH_TABLE_SIZE - 1);
  return (__brev(key) + (key^0xc375)) & (HASH_TABLE_SIZE - 1);
}

inline __device__ uint8_t encodePair(const uint8_t t1, const uint8_t t2)
{
  return ((t1 & 0x0f) << 4) | (t2 & 0x0f);
}

inline __device__ token_type decodePair(const uint8_t num)
{
  return token_type{static_cast<uint8_t>((num & 0xf0) >> 4),
                    static_cast<uint8_t>(num & 0x0f)};
}

inline __device__ void copyLiterals(
    uint8_t* const dest, const uint8_t* const source, const size_t length)
{
  for (size_t i = 0; i < length; ++i) {
    dest[i] = source[i];
  }
}

inline __device__ position_type lengthOfMatch(
    const uint8_t* const data,
    const position_type prev_location,
    const position_type next_location,
    const position_type length)
{
  assert(prev_location < next_location);


  position_type i;
  for (i = 0; i + next_location + 5 < length; ++i) {
    if (data[prev_location + i] != data[next_location + i]) {
      break;
    }
  }
  return i;
}

inline __device__ position_type
convertIdx(const offset_type offset, const position_type pos)
{
  constexpr const position_type OFFSET_SIZE = MAX_OFFSET + 1;

  assert(offset <= pos);

  position_type realPos = (pos / OFFSET_SIZE) * OFFSET_SIZE + offset;
  if (realPos >= pos) {
    realPos -= OFFSET_SIZE;
  }
  assert(realPos < pos);

  return realPos;
}

inline __device__ bool isValidHash(
    const uint8_t* const data,
    const offset_type* const hashTable,
    const position_type key,
    const position_type hashPos,
    const position_type decomp_idx)
{
  if (hashTable[hashPos] == NULL_OFFSET) {
    return false;
  }

  const position_type offset = convertIdx(hashTable[hashPos], decomp_idx);

  if (decomp_idx - offset > MAX_OFFSET) {
    // the offset can be up to 2^16-1, but the converted idx can be up to 2^16,
    // so we need to eliminate this case.
    return false;
  }

  const word_type hashKey = readWord<word_type>(data + offset);

  if (hashKey != key) {
    return false;
  }

  return true;
}

inline __device__ void writeSequenceData(
    uint8_t* const compData,
    const uint8_t* const decompData,
    const token_type token,
    const offset_type offset,
    const position_type decomp_idx,
    position_type& comp_idx)
{
  assert(token.num_matches == 0 || token.num_matches >= 4);

  // -> add token
  compData[comp_idx]
      = encodePair(token.numLiteralsForHeader(), token.numMatchesForHeader());
  ++comp_idx;

  // -> add literal length
  const position_type literalEncodingLength = token.lengthOfLiteralEncoding();
  if (literalEncodingLength) {
    writeLSIC(compData + comp_idx, token.numLiteralsOverflow());
    comp_idx += literalEncodingLength;
  }

  // -> add literals
  copyLiterals(
      compData + comp_idx, decompData + decomp_idx, token.num_literals);
  comp_idx += token.num_literals;

  // -> add offset
  if (token.num_matches > 0) {
    assert(offset > 0);

    writeWord(compData + comp_idx, offset);
    comp_idx += sizeof(offset);

    // -> add match length
    if (token.hasNumMatchesOverflow()) {
      writeLSIC(compData + comp_idx, token.numMatchesOverflow());
      comp_idx += token.lengthOfMatchEncoding();
    }
  }
}

__device__ void compressStream(
    uint8_t* compData,
    const uint8_t* decompData,
    size_t length,
    size_t* comp_length)
{
  position_type decomp_idx = 0;
  position_type comp_idx = 0;

  __shared__ offset_type hashTable[HASH_TABLE_SIZE];

  // fill hash-table with null-entries
  for (position_type i = threadIdx.x; i < HASH_TABLE_SIZE; i += blockDim.x) {
    hashTable[i] = NULL_OFFSET;
  }

  while (decomp_idx < length) {
    const position_type tokenStart = decomp_idx;
    while (true) {
      // begin adding tokens to the hash table until we find a match
      const word_type next = readWord<word_type>(decompData + decomp_idx);
      const position_type pos = decomp_idx;
      position_type hashPos = hash(next);

      if (decomp_idx + 5 + 4 >= length) {
        // jump to end
        decomp_idx = length;

        // no match -- literals to the end
        token_type tok;
        tok.num_literals = length - tokenStart;
        tok.num_matches = 0;
        writeSequenceData(compData, decompData, tok, 0, tokenStart, comp_idx);
        break;
      } else if (isValidHash(decompData, hashTable, next, hashPos, pos)) {
        token_type tok;
        const position_type match_location
            = convertIdx(hashTable[hashPos], pos);
        assert(match_location < decomp_idx);
        assert(decomp_idx - match_location <= MAX_OFFSET);

        // we found a match
        const offset_type match_offset = decomp_idx - match_location;
        assert(match_offset > 0);
        assert(match_offset <= decomp_idx);
        const position_type num_literals = pos - tokenStart;

        // compute match length
        const position_type num_matches
            = lengthOfMatch(decompData, match_location, pos, length);
        decomp_idx += num_matches;

        // -> write our token and literal length
        tok.num_literals = num_literals;
        tok.num_matches = num_matches;
        writeSequenceData(
            compData, decompData, tok, match_offset, tokenStart, comp_idx);

        break;
      } else if (decomp_idx + 12 < length) {
        // last match cannot be within 12 bytes of the end

        // TODO: we should overwrite matches in our hash table too, as they
        // are more recent

        // add it to our literals and dictionary
        hashTable[hashPos] = pos & MAX_OFFSET;
      }
      ++decomp_idx;
    }
  }

  *comp_length = comp_idx;
}

inline __device__ void decompressStream(
    uint8_t* buffer,
    uint8_t* decompData,
    const uint8_t* compData,
    const position_type comp_start,
    position_type length,
    block_stats_st* stats)
{
#ifdef LOG_CTA_CYCLES
  uint64_t start_clock;
  if (threadIdx.x == 0) {
    start_clock = clock64();
  }
#endif

  position_type comp_end = length + comp_start;

  BufferControl ctrl(buffer, compData, comp_end);
  ctrl.loadAt(comp_start);

  position_type decomp_idx = 0;
  position_type comp_idx = comp_start;
  while (comp_idx < comp_end) {
    if (comp_idx + PREFETCH_DIST > ctrl.end()) {
      ctrl.loadAt(comp_idx);
    }

    // read header byte
    token_type tok = decodePair(*ctrl.rawAt(comp_idx));
    ++comp_idx;

    // read the length of the literals
    position_type num_literals = tok.num_literals;
    if (tok.num_literals == 15) {
      num_literals += ctrl.readLSIC(comp_idx);
    }
#ifdef LOG_STATS
    if (threadIdx.x == 0) {
      atomicMin(&stats->copy_length_min, num_literals);
      atomicMax(&stats->copy_length_max, num_literals);
      atomicAdd(&stats->copy_length_sum, num_literals);
      if (tok.num_literals == 15) {
        atomicAdd(&stats->copy_lsic_count, 1);
      }
      atomicAdd(&stats->copy_length_count, 1);
    }
#endif
    const position_type literalStart = comp_idx;


    // copy the literals to the out stream
    if (num_literals + comp_idx > ctrl.end()) {
      coopCopyNoOverlap(
          decompData + decomp_idx, compData + comp_idx, num_literals);
    } else {
      // our buffer can copy
      coopCopyNoOverlap(
          decompData + decomp_idx, ctrl.rawAt(comp_idx), num_literals);
    }

    comp_idx += num_literals;
    decomp_idx += num_literals;

    // Note that the last sequence stops right after literals field.
    // There are specific parsing rules to respect to be compatible with the
    // reference decoder : 1) The last 5 bytes are always literals 2) The last
    // match cannot start within the last 12 bytes Consequently, a file with
    // less then 13 bytes can only be represented as literals These rules are in
    // place to benefit speed and ensure buffer limits are never crossed.
    if (comp_idx < comp_end) {

      // read the offset
      offset_type offset;
      if (comp_idx + sizeof(offset_type) > ctrl.end()) {
        offset = readWord<offset_type>(compData + comp_idx);
      } else {
        offset = readWord<offset_type>(ctrl.rawAt(comp_idx));
      }

      comp_idx += sizeof(offset_type);

      // read the match length
      position_type match = 4 + tok.num_matches;
      if (tok.num_matches == 15) {
        match += ctrl.readLSIC(comp_idx);
      }

#ifdef LOG_STATS
      if (threadIdx.x == 0) {
        atomicMin(&stats->match_length_min, match);
        atomicMax(&stats->match_length_max, match);
        atomicAdd(&stats->match_length_sum, match);
        atomicAdd(&stats->match_length_count, 1);
        if (tok.num_matches == 15) {
          atomicAdd(&stats->match_lsic_count, 1);
        }
        if (offset < match)
          atomicAdd(&stats->match_overlaps, 1);
        atomicMin(&stats->offset_min, offset);
        atomicMax(&stats->offset_max, offset);
        atomicAdd(&stats->offset_sum, offset);
      }
#endif

      // copy match
      if (offset <= num_literals
          && (ctrl.begin() <= literalStart
              && ctrl.end() >= literalStart + num_literals)) {
        // we are using literals already present in our buffer

        coopCopyOverlap(
            decompData + decomp_idx,
            ctrl.rawAt(literalStart + (num_literals - offset)),
            offset,
            match);
        // we need to sync after we copy since we use the buffer
        syncCTA();
      } else {
        // we need to sync before we copy since we use decomp
        syncCTA();

        coopCopyOverlap(
            decompData + decomp_idx,
            decompData + decomp_idx - offset,
            offset,
            match);
      }
      decomp_idx += match;
    }
  }
#ifdef LOG_CTA_CYCLES
  if (threadIdx.x == 0)
    stats->cycles = clock64() - start_clock;
#endif
  assert(comp_idx == comp_end);
}


__global__ void lz4CompressMultistreamKernel(
    uint8_t* compData,
    const uint8_t* decompData,
    size_t chunk_size,
    size_t stride,
    size_t last_chunk_size,
    size_t* comp_length,
    size_t batch_bytes)
{
  const uint8_t* decomp_ptr = &decompData[blockIdx.x*chunk_size];
  uint8_t* comp_ptr = &compData[blockIdx.x*(stride)];  

  size_t decomp_length = chunk_size;
  if(blockIdx.x == gridDim.x-1 && last_chunk_size != 0) {
    decomp_length = last_chunk_size;
  }

  compressStream(
      comp_ptr,
      decomp_ptr,
      decomp_length,
      comp_length + blockIdx.x);

}


__global__ void copyToContig(
   void* compData,
   void* tempData,
   int stride,
   size_t* prefix_output,
   size_t* metadata_ptr)
{
  for(size_t i=threadIdx.x; i<(prefix_output[blockIdx.x+1] - prefix_output[blockIdx.x]); i+=blockDim.x) {
    ((uint8_t*)compData)[prefix_output[blockIdx.x] + i] = ((uint8_t*)tempData)[blockIdx.x*stride + i];
  }

__syncthreads();
if(threadIdx.x==0) {
  metadata_ptr[blockIdx.x] = prefix_output[blockIdx.x];
  metadata_ptr[blockIdx.x+1] = prefix_output[blockIdx.x+1];
}

}

__global__ void lz4DecompressMultistreamKernel(
    uint8_t* decompData,
    const uint8_t* compData,
    size_t* offsets,
    size_t decomp_chunk_size,
    size_t last_decomp_chunk_size,
    int num_chunks,
    block_stats_st* stats)
{
  const int bid = blockIdx.x * Y_DIM + threadIdx.y;

  __shared__ uint8_t buffer[BUFFER_SIZE * Y_DIM];

  if (bid < num_chunks) {
    uint8_t* decomp_ptr = &(decompData[bid * decomp_chunk_size]);
    size_t chunk_length = offsets[bid + 1] - offsets[bid];

    if (bid == num_chunks - 1 && last_decomp_chunk_size != 0)
      decomp_chunk_size = last_decomp_chunk_size;

    decompressStream(
        buffer + threadIdx.y * BUFFER_SIZE,
        decomp_ptr,
        compData,
        offsets[bid],
        chunk_length,
        stats + bid);
  }
}

/******************************************************************************
 * PUBLIC FUNCTIONS ***********************************************************
 *****************************************************************************/

void lz4CompressBatch(
    void* const compData,
    void* const tempData,
    const size_t temp_bytes,
    const uint8_t* decomp_ptr,
    uint8_t* metadata_ptr,
    size_t batch_bytes,
    int chunk_bytes,
    int chunks_in_batch,
    int blocks,
    cudaStream_t stream)
{
  const size_t stride = lz4ComputeMaxSize(chunk_bytes);

  TempSpaceBroker broker(tempData, temp_bytes);

  uint8_t* multiStreamTempSpace;
  broker.reserve(&multiStreamTempSpace, chunks_in_batch * stride);

  lz4CompressMultistreamKernel<<<blocks, 1, 0, stream>>>(
      multiStreamTempSpace,
      decomp_ptr,
      chunk_bytes,
      stride,
      batch_bytes % (chunk_bytes),
      reinterpret_cast<size_t*>(metadata_ptr),
      batch_bytes);

  size_t prefix_temp_size=0;

  size_t* prefix_out;
  broker.reserve(&prefix_out, chunks_in_batch + 1);

  // Compute exact temp space needed by cub
  cudaError_t err = cub::DeviceScan::InclusiveSum(
      NULL,
      prefix_temp_size,
      (((size_t*)metadata_ptr) - 1),
      prefix_out,
      (size_t)chunks_in_batch + 1,
      stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to get inclusvie some temp space requirements: "
        + std::to_string(err));
  }

  uint8_t* prefix_temp_storage;
  broker.reserve(&prefix_temp_storage, prefix_temp_size);

  err = cub::DeviceScan::InclusiveSum(
      prefix_temp_storage,
      prefix_temp_size,
      (((size_t*)metadata_ptr) - 1),
      prefix_out,
      (size_t)chunks_in_batch + 1,
      stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to launch inclusive sum: " + std::to_string(err));
  }
  // Copy prefix sums values to metadata header and copy compressed data into
  // contiguous space
  copyToContig<<<chunks_in_batch, 32, 0, stream>>>(
      compData,
      multiStreamTempSpace,
      stride,
      prefix_out,
      ((size_t*)metadata_ptr) - 1);
}


void lz4DecompressBatch( 
    void* decompData,
    const void* compData,
    int headerOffset,
    int chunk_size,
    int last_chunk_size,
    int chunks_in_batch,
    cudaStream_t stream)
{

  lz4DecompressMultistreamKernel<<<
      ((chunks_in_batch - 1) / Y_DIM)+1,
      dim3(DECOMP_THREADS, Y_DIM, 1),
      0, 
      stream>>> 
      ((uint8_t*)decompData, 
      ((uint8_t*)compData),
      (size_t*)(((uint8_t*)compData)+headerOffset), 
      chunk_size, 
      last_chunk_size,
      chunks_in_batch,
      NULL);

}

size_t lz4ComputeTempSize(const size_t maxChunksInBatch, const size_t chunkSize)
{
  size_t prefix_temp_size;
  cudaError_t err = cub::DeviceScan::InclusiveSum(
      NULL,
      prefix_temp_size,
      static_cast<const size_t*>(nullptr),
      static_cast<size_t*>(nullptr),
      maxChunksInBatch + 1);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to get space for cub inclusive sub: " + std::to_string(err));
  }

  const size_t strideSize = lz4ComputeMaxSize(chunkSize);
  const size_t prefix_out_size = sizeof(size_t) * (maxChunksInBatch + 1);

  return prefix_temp_size + prefix_out_size + strideSize * maxChunksInBatch;
}

size_t lz4ComputeMaxSize(const size_t size)
{
  const size_t expansion = size + 1 + roundUpDiv(size, 255);
  return roundUpTo(expansion, sizeof(size_t));
}

} // nvcomp namespace

