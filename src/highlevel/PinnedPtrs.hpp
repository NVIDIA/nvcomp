#pragma once

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>
#include <vector>

namespace nvcomp {

// Static values that should be exposed.
// These could be static members of the PinnedPtrPool 
// but it's complicated by PinnedPtrPool being a template class
static const int PINNED_POOL_PREALLOC_SIZE = 10;
static const int PINNED_POOL_REALLOC_SIZE = 5;

/** 
 * @brief A memory pool that can allocate pinned host memory in batches 
 * 
 * This class is able to allocate a number of members of type T at once. In standard
 * memory pool fashion, when the user is finished with a value, 
 * the pointer to the value is pushed back into the pool.
 * 
 */ 
template<typename T>
struct PinnedPtrPool {

private: // data
  std::vector<T*> alloced_buffers; 
  std::vector<T*> pool;

public: // members

  PinnedPtrPool() 
    : alloced_buffers(1),
      pool()
  {
    T*& first_alloc = alloced_buffers[0];

    pool.reserve(PINNED_POOL_PREALLOC_SIZE);

    gpuErrchk(cudaHostAlloc(&first_alloc, PINNED_POOL_PREALLOC_SIZE * sizeof(T), cudaHostAllocDefault));

    for (int ix = 0; ix < PINNED_POOL_PREALLOC_SIZE; ++ix) {
      pool.push_back(first_alloc + ix);
    }
  }

  /**
   * @brief Push the pointer back into the pool
   */ 
  void push_ptr(T* status) 
  {
    pool.push_back(status);
  }

  /**
   * @brief Get the number of available pointers without additional allocations
   */ 
  size_t get_current_pool_size() {
    return pool.size();
  }

  /**
   * @brief Get the total number of T instances that have been allocated
   */ 
  size_t get_alloced_size() {
    return (alloced_buffers.size() - 1) * PINNED_POOL_REALLOC_SIZE + PINNED_POOL_PREALLOC_SIZE;
  }

  /**
   * @brief Get a pointer to a T instance in pinned host memory from the pool
   */ 
  T* pop_ptr() 
  {
    if (pool.empty()) {
      // realloc
      alloced_buffers.push_back(nullptr);
      T*& new_alloc = alloced_buffers.back();

      gpuErrchk(cudaHostAlloc(&new_alloc, PINNED_POOL_REALLOC_SIZE * sizeof(T), cudaHostAllocDefault));
      for (int ix = 0; ix < PINNED_POOL_REALLOC_SIZE; ++ix) {
        get_current_pool_size();
        pool.push_back(new_alloc + ix);
      }
    } 
    T* res = pool.back();
    get_current_pool_size();
    pool.pop_back();
    get_current_pool_size();
    return res;
  }

  ~PinnedPtrPool() {
    for (auto alloced_buffer : alloced_buffers) {
      gpuErrchk(cudaFreeHost(alloced_buffer));
    }
  }
};

/** 
 * @brief A wrapper for pinned ptrs, interacts with PinnedPtrPool.
 * 
 * This class is intended to be held in a std::shared_ptr. Then, when the 
 * user is finished, the destructor automatically returns the underlying memory
 * to the PinnedPtrPool.
 * 
 */ 
template<typename T>
struct PinnedPtrWrapper {
  PinnedPtrPool<T>& memory_pool;
  T* ptr;

  /**
   * @brief The constructor gets a wrapped ptr from the memory pool
   */ 
  PinnedPtrWrapper(PinnedPtrPool<T>& memory_pool) 
    : memory_pool(memory_pool),
      ptr(memory_pool.pop_ptr())
  {}

  // Disallow copies
  PinnedPtrWrapper& operator=(const PinnedPtrWrapper&) = delete;
  PinnedPtrWrapper(const PinnedPtrWrapper&) = delete;

  /**
   * @brief Move constructor that steals the pointer from the expiring `other`
   */ 
  PinnedPtrWrapper(PinnedPtrWrapper&& other) 
    : memory_pool(other.memory_pool),
      ptr(other.ptr)
  {
    other.ptr = nullptr;
  }

  /**
   * @brief The destructor will automatically return the ptr to the memory pool
   */ 
  ~PinnedPtrWrapper() {
    if (ptr != nullptr) {
      memory_pool.push_ptr(ptr);
    }
  }
  
  /**
   * @brief Gets a reference to the underlying value
   */ 
  T& operator*() {
    return *ptr;
  }
};

} // namespace nvcomp