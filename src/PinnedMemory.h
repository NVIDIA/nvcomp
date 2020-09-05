/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVCOMP_HOSTMEMORY_H
#define NVCOMP_HOSTMEMORY_H

#include "cuda_runtime.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace nvcomp
{

template <typename T>
class PinnedMemory
{
public:
  PinnedMemory() : m_ptr(nullptr), m_size(0)
  {
    // do nothing
  }

  PinnedMemory(const size_t size) : PinnedMemory()
  {
    m_size = size;

    const size_t num_bytes = sizeof(T) * m_size;

    cudaError_t err = cudaMallocHost((void**)&m_ptr, num_bytes);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate " + std::to_string(num_bytes)
          + " bytes of pinned memory: " + std::to_string(err) + " : "
          + std::string(cudaGetErrorString(err)));
    }
  }

  template <typename ITER>
  PinnedMemory(ITER begin, ITER end) : PinnedMemory(end - begin)
  {
    std::copy(begin, end, m_ptr);
  }

  PinnedMemory(PinnedMemory&& other) : m_ptr(other.m_ptr), m_size(other.m_size)
  {
    other.m_ptr = nullptr;
    other.m_size = 0;
  }

  PinnedMemory& operator=(PinnedMemory&& other)
  {
    std::swap(m_ptr, other.m_ptr);
    std::swap(m_size, other.m_size);

    other.clear();

    return *this;
  }

  // deleted constructors
  PinnedMemory(const PinnedMemory& other) = delete;
  PinnedMemory& operator=(const PinnedMemory& other) = delete;

  ~PinnedMemory()
  {
    clear();
  }

  operator bool() const
  {
    return m_ptr != nullptr;
  }

  T& operator[](const size_t index)
  {
    return m_ptr[index];
  }

  const T& operator[](const size_t index) const
  {
    return m_ptr[index];
  }

  const T* begin() const
  {
    return m_ptr;
  }

  T* begin()
  {
    return m_ptr;
  }

  const T* end() const
  {
    return m_ptr + m_size;
  }

  T* end()
  {
    return m_ptr + m_size;
  }

  T* data()
  {
    return m_ptr;
  }

  const T* data() const
  {
    return m_ptr;
  }

  size_t size() const
  {
    return m_size;
  }

  void clear()
  {
    if (m_ptr) {
      cudaFreeHost(m_ptr);
      m_ptr = nullptr;
    }
    m_size = 0;
  }

private:
  T* m_ptr;
  size_t m_size;
};

} // namespace nvcomp

#endif
