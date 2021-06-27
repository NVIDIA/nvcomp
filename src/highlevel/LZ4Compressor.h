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

#pragma once

#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nvcomp
{
namespace highlevel
{

class LZ4Compressor
{
public:
  static size_t
  calculate_workspace_size(size_t decomp_data_size, size_t chunk_size);

  static size_t
  calculate_max_output_size(size_t decomp_data_size, size_t chunk_size);

  /**
   * @brief Create a new LZ4Compressor.
   *
   * @param decomp_data The data to compress.
   * @param decomp_data_size The size of the data to compress.
   * @param chunk_size The size of each chunk to compress.
   * @param data_type The type of the data to compress.
   */
  LZ4Compressor(
      const uint8_t* decomp_data, size_t decomp_data_size, size_t chunk_size, nvcompType_t data_type);

  LZ4Compressor(const LZ4Compressor& other) = delete;
  LZ4Compressor& operator=(const LZ4Compressor& other) = delete;

  /**
   * @brief Get the size of the workspace required.
   *
   * @return The size of the workspace in bytes.
   */
  size_t get_workspace_size() const;

  size_t get_max_output_size() const;

  /**
   * @brief Set the allocated workspace.
   *
   * @param workspace The workspace.
   * @param size The size of the workspace in bytes.
   */
  void configure_workspace(void* workspace, size_t size);

  void configure_output(uint8_t* device_location, size_t* device_offsets);

  void compress_async(cudaStream_t stream);

private:
  const uint8_t* m_input_ptr;
  size_t m_input_size;
  size_t m_chunk_size;
  nvcompType_t m_data_type;
  size_t m_num_chunks;
  uint8_t* m_output_ptr;
  size_t* m_output_offsets;
  void* m_workspace;
  size_t m_workspace_size;

  bool is_workspace_configured() const;

  bool is_output_configured() const;
};

} // namespace highlevel
} // namespace nvcomp
