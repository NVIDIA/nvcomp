/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "lz4.h"

#include "LZ4CompressionKernels.h"
#include "LZ4Metadata.h"
#include "common.h"
#include "nvcomp.h"
#include "nvcomp.hpp"
#include "type_macros.h"

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>


using namespace nvcomp;

/**************************************************************************************
 *     C-style API calls for BATCHED compression/decompress defined below.
 *****************************)*********************************************************/

nvcompError_t nvcompBatchedLZ4DecompressGetMetadata(
    const void** /*in_ptr*/,
    size_t* /*in_bytes*/,
    size_t /*batch_size*/,
    void** /*metadata_ptr*/,
    cudaStream_t /*stream*/)
{
  return nvcompErrorNotSupported;
}

void nvcompBatchedLZ4DecompressDestroyMetadata(void* /*metadata_ptr*/)
{
//  ::operator delete(metadata_ptr);
}

nvcompError_t
nvcompBatchedLZ4DecompressGetTempSize(const void* /*metadata_ptr*/, size_t* /*temp_bytes*/)
{
  return nvcompErrorNotSupported;
/*
  if (temp_bytes == NULL) {
    std::cerr << "Invalid, temp_bytes ptr NULL." << std::endl;
    return nvcompErrorInvalidValue;
  }
  // LZ4 decompression doesn't need any temp memory
  *temp_bytes = 0;

  if (metadata_ptr == NULL) {
    std::cerr << "Invalid, metadata ptr NULL." << std::endl;
    return nvcompErrorInvalidValue;
  }

  return nvcompSuccess;
*/
}

nvcompError_t
nvcompBatchedLZ4DecompressGetOutputSize(const void* /*metadata_ptr*/, size_t /*batch_size*/, size_t* /*output_bytes*/)
{
  return nvcompErrorNotSupported;
}

nvcompError_t nvcompBatchedLZ4DecompressAsync(
    const void* const* /*in_ptr*/,
    const size_t* /*in_bytes*/,
    size_t /*batch_size*/,
    void* const /*temp_ptr*/,
    const size_t /*temp_bytes*/,
    const void* const /*metadata_ptr*/,
    void* const /*out_ptr*/,
    const size_t* /*out_bytes*/,
    cudaStream_t /*stream*/)
{

  return nvcompErrorNotSupported;
/*
  if (metadata_ptr == NULL) {
    std::cerr << "Invalid, metadata NULL." << std::endl;
    return nvcompErrorInvalidValue;
  }
  return nvcompSuccess;
*/
}

nvcompError_t nvcompBatchedLZ4CompressGetTempSize(
    const void* const* const in_ptr,
    const size_t* const in_bytes,
    const size_t batch_size,
    const nvcompLZ4FormatOpts* const format_opts,
    size_t* const temp_bytes)
{
  try {
    // error check inputs
    if (format_opts == nullptr) {
      throw std::runtime_error("format_opts must not be null.");
    } else if (in_ptr == nullptr) {
      throw std::runtime_error("in_ptr must not be null.");
    } else if (in_bytes == nullptr) {
      throw std::runtime_error("in_bytes must not be null.");
    } else if (temp_bytes == nullptr) {
      throw std::runtime_error("temp_bytes must not be null.");
    }

    size_t max_temp_bytes = 0;
    for (size_t b = 0; b < batch_size; ++b) {
      if (in_ptr[b] == nullptr) {
        throw std::runtime_error(
            "in_ptr[" + std::to_string(b) + "] must not be null.");
      }

      size_t this_temp_bytes;
      nvcompError_t err = nvcompLZ4CompressGetTempSize(
          in_ptr[b],
          in_bytes[b],
          NVCOMP_TYPE_BITS,
          format_opts,
          &this_temp_bytes);
      if (err != nvcompSuccess) {
        throw std::runtime_error(
            "Error getting temp size for batch item " + std::to_string(b)
            + " : " + std::to_string(err));
      }

      if (this_temp_bytes > max_temp_bytes) {
        max_temp_bytes = this_temp_bytes;
      }
    }

    *temp_bytes = max_temp_bytes;
  } catch (const std::exception& e) {
    std::cerr << "Failed to get temp size for batch: " << e.what() << std::endl;
    return nvcompErrorCudaError;
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedLZ4CompressGetOutputSize(
    const void* const* const in_ptr,
    const size_t* const in_bytes,
    const size_t batch_size,
    const nvcompLZ4FormatOpts* const format_opts,
    void* const temp_ptr,
    const size_t temp_bytes,
    size_t* const out_bytes)
{
  try {
    // error check inputs
    if (format_opts == nullptr) {
      throw std::runtime_error("format_opts must not be null.");
    } else if (in_ptr == nullptr) {
      throw std::runtime_error("in_ptr must not be null.");
    } else if (in_bytes == nullptr) {
      throw std::runtime_error("in_bytes must not be null.");
    } else if (temp_ptr == nullptr) {
      throw std::runtime_error("temp_ptr must not be null.");
    } else if (out_bytes == nullptr) {
      throw std::runtime_error("out_bytes must not be null.");
    }

    for (size_t b = 0; b < batch_size; ++b) {
      if (in_ptr[b] == nullptr) {
        throw std::runtime_error(
            "in_ptr[" + std::to_string(b) + "] must not be null.");
      }

      nvcompError_t err = nvcompLZ4CompressGetOutputSize(
          in_ptr[b],
          in_bytes[b],
          NVCOMP_TYPE_BITS,
          format_opts,
          temp_ptr,
          temp_bytes,
          out_bytes + b,
          false);
      if (err != nvcompSuccess) {
        throw std::runtime_error(
            "Error getting output size for batch item " + std::to_string(b)
            + " : " + std::to_string(err));
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "Failed to get output size for batch: " << e.what()
              << std::endl;
    return nvcompErrorCudaError;
  }

  return nvcompSuccess;
}

nvcompError_t nvcompBatchedLZ4CompressAsync(
    const void* const* const in_ptr,
    const size_t* const in_bytes,
    const size_t batch_size,
    const nvcompLZ4FormatOpts* const format_opts,
    void* const temp_ptr,
    size_t const temp_bytes,
    void* const* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  try {
    // error check inputs
    if (format_opts == nullptr) {
      throw std::runtime_error("format_opts must not be null.");
    } else if (in_ptr == nullptr) {
      throw std::runtime_error("in_ptr must not be null.");
    } else if (in_bytes == nullptr) {
      throw std::runtime_error("in_bytes must not be null.");
    } else if (temp_ptr == nullptr) {
      throw std::runtime_error("temp_ptr must not be null.");
    } else if (out_ptr == nullptr) {
      throw std::runtime_error("out_ptr must not be null.");
    } else if (out_bytes == nullptr) {
      throw std::runtime_error("out_bytes must not be null.");
    }

    for (size_t b = 0; b < batch_size; ++b) {
      if (in_ptr[b] == nullptr) {
        throw std::runtime_error(
            "in_ptr[" + std::to_string(b) + "] must not be null.");
      } else if (out_ptr[b] == nullptr) {
        throw std::runtime_error(
            "out_ptr[" + std::to_string(b) + "] must not be null.");
      }

      nvcompError_t err = nvcompLZ4CompressAsync(
          in_ptr[b],
          in_bytes[b],
          NVCOMP_TYPE_BITS,
          format_opts,
          temp_ptr,
          temp_bytes,
          out_ptr[b],
          out_bytes + b,
          stream);
      if (err != nvcompSuccess) {
        throw std::runtime_error(
            "Error launching compression batch item " + std::to_string(b)
            + " : " + std::to_string(err));
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "Failed launch compression for batch: " << e.what()
              << std::endl;
    return nvcompErrorCudaError;
  }

  return nvcompSuccess;
}
