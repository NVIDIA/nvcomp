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

#ifndef NVCOMP_CASCADED_HPP
#define NVCOMP_CASCADED_HPP

#include "cascaded.h"
#include "nvcomp.hpp"

#include <cuda_runtime.h>

namespace nvcomp
{

/**
 * @brief Primary compressor offered by nvcomp: RLE-Delta w/ bit-packing
 * Compression and decompression run asynchronously, but compress() requires
 * that the compressed size (*out_btyes) is known and buffers allocated. Can
 * define synchronous wrapper that includes size estimation kernel + allocation.
 */
class CascadedCompressor : public Compressor
{
public:
  /**
   * @brief Create a new CascadedCompressor.
   *
   * NOTE: Currently, cascaded compression is limited to 2^31-1 bytes. To
   * compress larger data, break it up into chunks.
   *
   * @param type The data type being compressed.
   * @param num_RLEs The number of Run Length encodings to perform.
   * @param num_deltas The number of Deltas to perform.
   * @param use_bp Whether or not to bitpack the end result.
   */
  CascadedCompressor(
      nvcompType_t type, int num_RLEs, int num_deltas, bool use_bp);

  /**
   * @brief Create a new CascadedCompressor without defining the configuration
   * (RLE, delta, bp). Runs the cascaded selector before compression to
   * determine the configuration.
   *
   * NOTE: This results in the compress_async() synchronizing with the input
   * stream.
   *
   * @param in_ptr The input data on the GPU to compress.
   * @param num_elements The number of elements to compress.
   */
  explicit CascadedCompressor(nvcompType_t type);

  // disable copying
  CascadedCompressor(const CascadedCompressor&) = delete;
  CascadedCompressor& operator=(const CascadedCompressor&) = delete;

  /**
   * @brief Configure the compressor for the given input, and get the necessary
   * spaces.
   *
   * @param in_bytes The size of the input in bytes.
   * @param temp_bytes The temporary workspace required (output).
   * @param out_bytes The maximum possible output size (output).
   */
  void configure(
      const size_t in_bytes, size_t* temp_bytes, size_t* out_bytes) override;

  /**
   * @brief Perform compression asynchronously.
   *
   * NOTE: If the the Cascded configuration was not specified, that is the
   * constructor wiht only one argument is used, then this method will
   * synchronize with the stream while determining the proper configuration to
   * run with. It will then launch compression asynchronously on the stream and
   * return.
   *
   * @param in_ptr The uncompressed input data (GPU accessible).
   * @param in_bytes The length of the uncompressed input data.
   * @param temp_ptr The temporary workspace (GPU accessible).
   * @param temp_bytes The size of the temporary workspace.
   * @param out_ptr The location to output data to (GPU accessible).
   * @param out_bytes The size of the output location on input, and the size of
   * the compressed data on output (CPU accessible, but must be pinned or
   * managed memory for this function to be asynchronous).
   * @param stream The stream to operate on.
   *
   * @throw NVCompException If compression fails to launch on the stream.
   */
  void compress_async(
      const void* in_ptr,
      const size_t in_bytes,
      void* temp_ptr,
      const size_t temp_bytes,
      void* out_ptr,
      size_t* out_bytes,
      cudaStream_t stream) override;

private:
  nvcompType_t m_type;
  nvcompCascadedFormatOpts m_opts;
};

class CascadedDecompressor : public Decompressor
{
public:
  CascadedDecompressor();

  ~CascadedDecompressor();

  // disable copying
  CascadedDecompressor(const CascadedDecompressor&) = delete;
  CascadedDecompressor& operator=(const CascadedDecompressor&) = delete;

  /**
   * @brief Configure the decompressor. This synchronizes with the stream.
   *
   * @param in_ptr The compressed data on the device.
   * @param in_bytes The size of the compressed data.
   * @param temp_bytes The temporary space required for decompression (output).
   * @param out_bytes The size of the uncompressed data (output).
   * @param stream The stream to operate on for copying data from the device to
   * the host.
   */
  void configure(
      const void* in_ptr,
      const size_t in_bytes,
      size_t* temp_bytes,
      size_t* out_bytes,
      cudaStream_t stream) override;

  /**
   * @brief Decompress the given data asynchronously.
   *
   * @param temp_ptr The temporary workspace on the device to use.
   * @param temp_bytes The size of the temporary workspace.
   * @param out_ptr The location to write the uncompressed data to on the
   * device.
   * @param out_num_elements The size of the output location in number of
   * elements.
   * @param stream The stream to operate on.
   *
   * @throw NVCompException If decompression fails to launch on the stream.
   */
  void decompress_async(
      const void* in_ptr,
      const size_t in_bytes,
      void* temp_ptr,
      const size_t temp_bytes,
      void* out_ptr,
      const size_t out_bytes,
      cudaStream_t stream) override;

private:
  void* m_metadata_ptr;
  size_t m_metadata_bytes;
};

/******************************************************************************
 * Cascaded Selector **********************************************************
 *****************************************************************************/
/**
 *@brief Primary class for the Cascaded Selector used to determine the
 * best configuration to run cascaded compression on a given input.
 *
 *@param T the datatype of the input
 */
template <typename T>
class CascadedSelector
{
private:
  const void* input_data;
  size_t input_byte_len;
  size_t max_temp_size; // Internal variable used to store the temp buffer size
  nvcompCascadedSelectorOpts opts; // Sampling options

public:
  /**
   *@brief Create a new CascadedSelector for the given input data
   *
   *@param input The input data device pointer to select a cheme for
   *@param byte_len The number of bytes of input data
   *@param num_sample_ele The number of elements in a sample
   *@param num_sample The number of samples
   *@param type The type of input data
   */
  CascadedSelector(
      const void* input, size_t byte_len, nvcompCascadedSelectorOpts opts);

  // disable copying
  CascadedSelector(const CascadedSelector&) = delete;
  CascadedSelector& operator=(const CascadedSelector&) = delete;

  /*
   *@brief return the required size of workspace buffer in bytes
   */
  size_t get_temp_size() const;

  /*
   *@brief Select a CascadedSelector compression scheme that can provide the
   *best compression ratio and reports estimated compression ratio.
   *
   *@param d_worksapce The device potiner for the workspace
   *@param workspace_len The size of workspace buffer in bytes
   *@param comp_ratio The estimated compssion ratio using the bbest scheme
   *(output)
   *@param stream The input stream to run the select function
   *@return Selected Cascaded options (RLE, Delta encoding, bit packing)
   */
  nvcompCascadedFormatOpts select_config(
      void* d_workspace,
      size_t workspace_len,
      double* comp_ratio,
      cudaStream_t stream);

  /*
   *@brief Select a CascadedSelector compression scheme that can provide the
   *best compression ratio - does NOT return estimated compression ratio.
   *
   *@param d_worksapce The device potiner for the workspace
   *@param workspace_len The size of workspace buffer in bytes
   *@param stream The input stream to run the select function
   *@return Selected Cascaded options (RLE, Delta encoding, bit packing)
   */
  nvcompCascadedFormatOpts
  select_config(void* d_workspace, size_t workspace_len, cudaStream_t stream);
};

/******************************************************************************
 * METHOD IMPLEMENTATIONS *****************************************************
 *****************************************************************************/

inline CascadedCompressor::CascadedCompressor(
    nvcompType_t type, int num_RLEs, int num_deltas, bool use_bp) :
    m_type(type),
    m_opts{num_RLEs, num_deltas, use_bp}
{
  // do nothing
}

inline CascadedCompressor::CascadedCompressor(nvcompType_t type) :
    CascadedCompressor(type, -1, -1, false)
{
  // do nothing
}

inline void CascadedCompressor::configure(
    const size_t in_bytes, size_t* const temp_bytes, size_t* const out_bytes)
{
  size_t metadata_bytes;
  nvcompCascadedFormatOpts* temp_opts = &m_opts;
  if(m_opts.num_RLEs == -1) {
    temp_opts = NULL;
  }

  nvcompError_t status = nvcompCascadedCompressConfigure(
        temp_opts, m_type, in_bytes, &metadata_bytes, temp_bytes, out_bytes);

  throwExceptionIfError(status, "nvcompCascadedCompressConfigure() failed");
}

inline void CascadedCompressor::compress_async(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{

  nvcompCascadedFormatOpts* temp_opts = &m_opts;
  if(m_opts.num_RLEs == -1) {
    temp_opts = NULL;
  }

  nvcompError_t status = nvcompCascadedCompressAsync(
      temp_opts,
      m_type,
      in_ptr,
      in_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      out_bytes,
      stream);
  throwExceptionIfError(status, "nvcompCascadedCompressAsync() failed");
}

inline CascadedDecompressor::CascadedDecompressor() :
    m_metadata_ptr(nullptr),
    m_metadata_bytes(0)
{
  // do nothing
}

inline CascadedDecompressor::~CascadedDecompressor()
{
  if (m_metadata_ptr) {
    nvcompCascadedDestroyMetadata(m_metadata_ptr);
  }
}

inline void CascadedDecompressor::configure(
    const void* const in_ptr,
    const size_t in_bytes,
    size_t* const temp_bytes,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  nvcompError_t status = nvcompCascadedDecompressConfigure(
      in_ptr,
      in_bytes,
      &m_metadata_ptr,
      &m_metadata_bytes,
      temp_bytes,
      out_bytes,
      stream);
  throwExceptionIfError(status, "nvcompCascadedConfigure() failed");
}

inline void CascadedDecompressor::decompress_async(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    const size_t out_bytes,
    cudaStream_t stream)
{
  nvcompError_t status = nvcompCascadedDecompressAsync(
      in_ptr,
      in_bytes,
      m_metadata_ptr,
      m_metadata_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      out_bytes,
      stream);
  throwExceptionIfError(status, "nvcompCascadedQeueryMetadataAsync() failed");
}

template <typename T>
inline CascadedSelector<T>::CascadedSelector(
    const void* input,
    const size_t byte_len,
    nvcompCascadedSelectorOpts selector_opts) :
    input_data(input),
    input_byte_len(byte_len),
    max_temp_size(0),
    opts(selector_opts)
{
  size_t temp;

  nvcompError_t status = nvcompCascadedSelectorConfigure(
      &opts, TypeOf<T>(), input_byte_len, &temp);
  throwExceptionIfError(status, "SelectorGetTempSize failed");

  this->max_temp_size = temp;
}

template <typename T>
inline size_t CascadedSelector<T>::get_temp_size() const

{
  return max_temp_size;
}

template <typename T>
inline nvcompCascadedFormatOpts CascadedSelector<T>::select_config(
    void* d_workspace,
    size_t workspace_size,
    double* comp_ratio,
    cudaStream_t stream)
{
  nvcompCascadedFormatOpts cascadedOpts;
  nvcompError_t status = nvcompCascadedSelectorRun(
      &opts,
      TypeOf<T>(),
      input_data,
      input_byte_len,
      d_workspace,
      workspace_size,
      &cascadedOpts,
      comp_ratio,
      stream);
  throwExceptionIfError(status, "SelectorSelectConfig failed");

  return cascadedOpts;
}

template <typename T>
inline nvcompCascadedFormatOpts CascadedSelector<T>::select_config(
    void* d_workspace, size_t workspace_size, cudaStream_t stream)
{
  double comp_ratio;
  return select_config(d_workspace, workspace_size, &comp_ratio, stream);
}

} // namespace nvcomp
#endif
