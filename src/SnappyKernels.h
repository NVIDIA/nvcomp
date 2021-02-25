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


namespace nvcomp {

/**
 * @brief Input parameters for the decompression interface
 **/
struct gpu_inflate_input_s {
  const void *srcDevice;
  uint64_t srcSize;
  void *dstDevice;
  uint64_t dstSize;
};

/**
 * @brief Output parameters for the decompression interface
 **/
struct gpu_inflate_status_s {
  uint64_t bytes_written;
  uint32_t status;
  uint32_t reserved;
};

/**
 * @brief Interface for compressing data with Snappy
 *
 * Multiple, independent chunks of compressed data can be compressed by using
 * separate gpu_inflate_input_s/gpu_inflate_status_s pairs for each chunk.
 *
 * @param[in] inputs List of input argument structures
 * @param[out] outputs List of output status structures
 * @param[in] count Number of input/output structures, default 1
 * @param[in] stream CUDA stream to use, default 0
 **/
cudaError_t gpu_snap(gpu_inflate_input_s *inputs,
                              gpu_inflate_status_s *outputs,
                              int count,
                              cudaStream_t stream);
}

