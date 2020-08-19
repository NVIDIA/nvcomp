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


namespace nvcomp {

// TODO - document headers
// Compress a single batch of chunks, placing result in contiguous space in compData
void lz4CompressBatch(
    void* compData,
    void* tempData,
    const size_t temp_bytes,
    const uint8_t* decomp_ptr,
    uint8_t* metadata_ptr,
    size_t batch_bytes,
    int chunk_bytes,
    int chunks_in_batch,
    int blocks,
    cudaStream_t stream);

// Decompress a single batch of chunks, placing the result in decompData
void lz4DecompressBatch(
    void* tempData,
    const size_t temp_bytes,
    void* decompData,
    const void* compData,
    int headerOffset,
    int chunk_size,
    int chunks_in_batch,
    cudaStream_t stream);

size_t lz4CompressComputeTempSize(
    const size_t max_chunks_in_batch, const size_t chunk_size);

size_t lz4DecompressComputeTempSize(
    const size_t max_chunks_in_batch, const size_t chunk_size);

size_t lz4ComputeMaxSize(const size_t chunk_size);
}

