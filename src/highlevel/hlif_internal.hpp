#pragma once

#include <stdint.h>

namespace nvcomp {

typedef uint64_t ChunkStartOffset_t;
typedef uint32_t Checksum_t;

struct CommonHeader {
  uint32_t magic_number; // 
  uint8_t major_version;
  uint8_t minor_version;
  uint64_t comp_data_size;
  uint64_t decomp_data_size;
  size_t num_chunks;
  bool include_chunk_starts;
  Checksum_t full_comp_buffer_checksum;
  Checksum_t decomp_buffer_checksum;
  bool include_per_chunk_comp_buffer_checksums;
  bool include_per_chunk_decomp_buffer_checksums;
  size_t uncomp_chunk_size;
  uint32_t comp_data_offset;
};

} // end namespace nvcomp