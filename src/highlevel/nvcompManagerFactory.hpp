// Construct a ManagerBase from a buffer

#include "nvcompManager.hpp"
#include "LZ4BatchManager.hpp"
#include "SnappyBatchManager.hpp"

namespace nvcomp {

std::shared_ptr<nvcompManagerBase> create_manager(const uint8_t* comp_buffer, cudaStream_t stream = 0, const int device_id = 0) {
  // Need to determine the type of manager
  const CommonHeader* common_header = reinterpret_cast<const CommonHeader*>(comp_buffer);
  CommonHeader cpu_common_header;
  gpuErrchk(cudaMemcpy(&cpu_common_header, common_header, sizeof(CommonHeader), cudaMemcpyDefault));

  std::shared_ptr<nvcompManagerBase> res;

  switch(cpu_common_header.format) {
    case FormatType::LZ4: 
    {
      LZ4FormatSpecHeader format_spec;
      const LZ4FormatSpecHeader* gpu_format_header = reinterpret_cast<const LZ4FormatSpecHeader*>(comp_buffer + sizeof(CommonHeader));
      gpuErrchk(cudaMemcpy(&format_spec, gpu_format_header, sizeof(LZ4FormatSpecHeader), cudaMemcpyDefault));

      res = std::make_shared<LZ4BatchManager>(cpu_common_header.uncomp_chunk_size, format_spec.data_type, stream, device_id);
      break;
    }
    case FormatType::Snappy: 
    {
      SnappyFormatSpecHeader format_spec;
      const SnappyFormatSpecHeader* gpu_format_header = reinterpret_cast<const SnappyFormatSpecHeader*>(comp_buffer + sizeof(CommonHeader));
      gpuErrchk(cudaMemcpy(&format_spec, gpu_format_header, sizeof(SnappyFormatSpecHeader), cudaMemcpyDefault));

      res = std::make_shared<SnappyBatchManager>(cpu_common_header.uncomp_chunk_size, stream, device_id);
      break;
    }
    case FormatType::GDeflate: 
    {
      // TODO
      break;
    }
    case FormatType::Bitcomp: 
    {
      // TODO
      break;
    }
    case FormatType::ANS: 
    {
      // TODO
      break;
    }
    case FormatType::Cascaded: 
    {
      // TODO
      break;
    }
  }

  return res;
}

} // namespace nvcomp 
 