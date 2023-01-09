#define CATCH_CONFIG_MAIN

#include "tests/catch.hpp"
#include <cuda_runtime.h>
#include "CudaUtils.h"
#include "CRC32.hpp"

#include <boost/crc.hpp>
#include <random>

using namespace nvcomp;
using namespace std;

// generate uniformly random data
// random bytes
// random chunk sizes
// returns max chunk size
void gen_random_data(
  void **uncomp_chunks, size_t *uncomp_chunk_sizes, std::vector<size_t>& host_uncomp_sizes, 
  std::vector<void*>& host_uncomp_chunks, 
  size_t num_chunks, size_t& chunk_size, size_t max_chunk_size, size_t& total_size_bytes,
  int seed) {
  size_t min_chunk_size = 1;

  std::mt19937 gen(seed);

  std::vector<uint8_t> uncomp_chunk_host(max_chunk_size);
  size_t max_actual_chunk_size = 0;
  total_size_bytes = 0;

  for (size_t i = 0; i < num_chunks; ++i) {
    void *uncomp_chunk_dev;

    // pick a random size for chunk
    size_t uncomp_chunk_size = min_chunk_size + gen() % (max_chunk_size - min_chunk_size);
    if(uncomp_chunk_size > max_actual_chunk_size) {
      max_actual_chunk_size = uncomp_chunk_size;
    }
    total_size_bytes += uncomp_chunk_size;
    // allocate device chunk
    CudaUtils::check(cudaMalloc(&uncomp_chunk_dev, uncomp_chunk_size));

    // create random chunk on host byte-by-byte
    for (size_t j = 0; j < uncomp_chunk_size; ++j) {
      uncomp_chunk_host[j] = gen();
    }

    // copy host chunk to device chunk
    CudaUtils::check(cudaMemcpy(uncomp_chunk_dev, uncomp_chunk_host.data(), uncomp_chunk_size, cudaMemcpyDefault));

    // set uncomp chunk size
    CudaUtils::check(cudaMemcpy(&uncomp_chunk_sizes[i], &uncomp_chunk_size, sizeof(size_t), cudaMemcpyDefault));

    // copy pointer to device chunk into uncomp chunk array
    CudaUtils::check(cudaMemcpy(&uncomp_chunks[i], &uncomp_chunk_dev, sizeof(void*), cudaMemcpyDefault));

    // store device pointer to each chunk to free later
    host_uncomp_chunks[i] = uncomp_chunk_dev;
    host_uncomp_sizes[i] = uncomp_chunk_size;
  }
  
  chunk_size = max_actual_chunk_size;
}
TEST_CASE("non-interleaved CRC validation") {
  cudaStream_t stream;
  CudaUtils::check(cudaStreamCreate(&stream));
  
  const size_t batch_size = 1024;
  const size_t chunk_size = 1024;

  void **uncomp_chunks;
  size_t *uncomp_chunk_sizes;
  uint32_t *result_crcs;
  CudaUtils::check(cudaMalloc(&uncomp_chunks, batch_size*sizeof(void*)));
  CudaUtils::check(cudaMalloc(&uncomp_chunk_sizes, batch_size*sizeof(size_t)));
  CudaUtils::check(cudaMalloc(&result_crcs, batch_size*sizeof(uint32_t)));

  std::vector<size_t> host_uncomp_sizes(batch_size);
  std::vector<void*> host_uncomp_chunks(batch_size);

  size_t max_actual_chunk_size, total_size_bytes;

  gen_random_data(
    uncomp_chunks, uncomp_chunk_sizes, host_uncomp_sizes, host_uncomp_chunks, batch_size, max_actual_chunk_size, chunk_size, 
    total_size_bytes, /*seed=*/12);
  
  // copy GPU data over to the host for verification
  std::vector<void*> host_host_uncomp_chunks(batch_size);
  for(size_t i = 0; i < batch_size; ++i) {
    host_host_uncomp_chunks[i] = malloc(host_uncomp_sizes[i]);
    CudaUtils::check(cudaMemcpy(host_host_uncomp_chunks[i], host_uncomp_chunks[i], host_uncomp_sizes[i], cudaMemcpyDefault));
  }
  
  compute_uncomp_chunk_checksums(batch_size, uncomp_chunks, uncomp_chunk_sizes, result_crcs);

  std::vector<uint32_t> host_result_crcs(batch_size);
  CudaUtils::check(cudaMemcpy(host_result_crcs.data(), result_crcs, batch_size*sizeof(uint32_t), cudaMemcpyDefault));

  size_t num_failures = 0;
  for(size_t i = 0; i < batch_size; ++i) {
    boost::crc_32_type ref_crc;
    ref_crc.process_bytes(host_host_uncomp_chunks[i], host_uncomp_sizes[i]);
    if(host_result_crcs[i] != ref_crc.checksum()) {
      num_failures++;
    }
  }
  REQUIRE(num_failures == 0);

  for(auto chunk : host_uncomp_chunks) {
    CudaUtils::check(cudaFree(chunk));
  }
  CudaUtils::check(cudaFree(uncomp_chunks));
  CudaUtils::check(cudaFree(uncomp_chunk_sizes));
  CudaUtils::check(cudaFree(result_crcs));
}