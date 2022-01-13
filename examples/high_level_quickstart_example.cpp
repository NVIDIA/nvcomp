#include <random>
#include <assert.h>
#include <iostream>

#include "src/highlevel/nvcompManager.hpp"
#include "src/highlevel/nvcompManagerFactory.hpp"

/* 
  To build, execute
  
  mkdir build
  cd build
  cmake -DBUILD_EXAMPLES=ON ..
  make -j

  To execute, 
  bin/high_level_quickstart_example
*/

using namespace nvcomp;

void execute_example(uint8_t* device_input_data, const size_t in_bytes)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  LZ4BatchManager nvcomp_manager{chunk_size, data_type, stream};
  CompressionConfig comp_config = nvcomp_manager.configure_compression(in_bytes);

  uint8_t* comp_buffer;
  cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size);
  
  nvcomp_manager.compress(device_input_data, in_bytes, comp_buffer, comp_config);

  // Construct a new nvcomp manager from the compressed buffer.
  // Note we could use the nvcomp_manager from above, but here we demonstrate how to create a manager 
  // for the use case where a buffer is received and the user doesn't know how it was compressed
  auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);
  
  DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
  uint8_t* res_decomp_buffer;
  cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size);

  decomp_nvcomp_manager->decompress(res_decomp_buffer, comp_buffer, decomp_config);

  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);
}

int main()
{
  // Initialize a random array of chars
  const size_t in_bytes = 1000000;
  std::vector<uint8_t> uncompressed_data(in_bytes);
  
  std::mt19937 random_gen(42);

  // char specialization of std::uniform_int_distribution is
  // non-standard, and isn't available on MSVC, so use short instead,
  // but with the range limited, and then cast below.
  std::uniform_int_distribution<short> uniform_dist(0, 255);
  for (size_t ix = 0; ix < in_bytes; ++ix) {
    uncompressed_data[ix] = static_cast<uint8_t>(uniform_dist(random_gen));
  }

  uint8_t* device_input_data;
  cudaMalloc((void**)&device_input_data, in_bytes);
  cudaMemcpy(device_input_data, uncompressed_data.data(), in_bytes, cudaMemcpyDefault);
  
  execute_example(device_input_data, in_bytes);

  cudaFree(device_input_data);

  return 0;
}