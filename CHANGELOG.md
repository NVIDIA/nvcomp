# nvcomp 3.0.0
## New features
  - Added `nvcomp*RequiredAlignment` constant variables for each compressor
  - Low-level batched functions now return `nvcompErrorAlignment` if device buffers aren't sufficiently aligned
  - Added HLIF for ZSTD, Deflate. Updated HLIF design such that HLIF now dispatches to LLIF.
  - Introduced device-side API. Currently limited to the ANS format
  - Added support for logging using `NVCOMP_LOG_LEVEL` (0-5) and `NVCOMP_LOG_FILE` environment variables.

## Performance Optimizations
  - Optimize zSTD decompression. Up to 2.2x faster on H100 and 1.5x faster on A100
  - Optimize LZ4 decompression. Up to 1.4x faster on H100 and 1.4x faster on A100.
  - Optimize Snappy decompression. Up to 1.3x faster on H100 and 1.9x faster on A100.
  - Optimize Bitcomp decompression (standard algo). Up to 2x faster and more consistent accross datasets
  - Improve ZSTD compression ratio by up to 5% on 64 KB chunks, 30% on 512 KB chunks to closely match CPU L1 Compression.
  
# nvcomp 2.6.1 (2023-2-3)
## Bug fixes
  - Fixed a bug that caused non-deterministic decompression accuracy failures in ZSTD
  - Added support for Ada (sm89) GPUs 
  - Fixed inconsistent compression stream format on some datasets when using GDeflate high-compression algorithm.

# nvcomp 2.6.0 (2023-1-16)
## New Features
  - Added new nvcompBatched*CompressGetTempSizeEx API to allow
  less pessimistic scratch allocation requirement in many cases.
  - Further reduced zstd compression scratch requirement. For 
  very large batches, in conjunction with the new extended API, 
  the scratch allocation is now ~1.5x the total uncompressed 
  size of the batch.

# nvcomp 2.5.1 (2023-1-9)
## Bug fixes
  - Improved GDeflate decompression throughput by up to 2x, fixing perf regression in 2.5.0
  - Fixed issue where some uses of CUB and Thrust in nvCOMP weren't namespaced
  - Fixed bug, introduced in 2.5.0, in ZSTD decompression of large frames produced by the CPU compressor

# nvcomp 2.5.0 (2022-12-16)
## New features
  - Added Standard CRC32 support and its LLAPI.
  - Added Gzip batched decompresssion LL APIs, include getting decompression size APIs.
  - Added independent bitcomp.h header to access full feature set of bitcomp compressor
  - Added doc directory in nvcomp package containing the documentation files
  - Increased zStandard maximum compression chunk size from 64 KB to 16 MB 
  - Improved zStandard decompression throughput by up to 2x on small batches and 40% on large batches
  - Added `nvcomp*CompressionMaxAllowedChunkSize` constant variables for each compressor
  - Updated GDeflate stream format to make it compatible with the GDeflate compression standard in NVIDIA RTX IO and Microsoft DirectStorage 1.1.
  - Updated GDeflate to support 64 KB dictionary window which allows a higher compression ratio.
  - Updated GDeflate CPU implementation to use the open source libdeflate repo: https://github.com/NVIDIA/libdeflate
  - Added initial support for SM90

## Bug fixes
  - Fixed memcheck failure in Snappy compression
  - Fixed deflate compression issue related to very small chunk sizes
  - Fixed handling of zero-byte chunks in ANS, Bitcomp, Cascaded, Deflate, and Gdeflate compressors
  - Fixed bug in Bitcomp where the maximum compressed size was slightly underestimated.

# nvcomp 2.4.1 (2022-10-06)
## New features
  - The Deflate batched decompression API can now accept nullptr for actual_decompressed_bytes.
## Bug fixes
  - Fixed incorrect behavior, failure, or crash when using duplicates feature (`-x <count>`) of the low-level "chunked" benchmarks.
  - Updated deflate_cpu_compression example to use the correct APIs.
  - The Deflate batched decompression API can work on uncomprressed data chunk larger than 64KB.
  - Fixed correctness / stability issue in compute capability 6.1

# nvcomp 2.4.0 (2022-09-23)
## New features
  - Added support for ZSTD compression to LL API
  - Early Access Linux SBSA binaries.
## Bug fixes
  - Fixed issue where cascaded compressor bitpack wasn't considering unsigned data type, causing suboptimal compression ratio
  - Fixed cmake problem where we stated wrong version compatibility

## Performance Optimizations
  - Optimized GDeflate high-compression mode. Up to 2x faster.
  - Optimized ZSTD decompression. Up to 1.2x faster.
  - Optimized Deflate decompression. Up to 1.5x faster.
  - Optimized ANS compression. Strong scaling allows for up to 7x higher compression and decompression
    throughput for files on the order of a few MB in size. Decompression throughput is improved by at least 
    20% on all tested files.
  
# nvcomp 2.3.3 (2022-07-20)
## Bug Fixes
  - Add missing nvcompBatchedDeflateDecompressGetTempSizeEx API
  - Fixed minor correctness issue in deflate compression. 
  - Fixed cmake problem that caused an unnecessary implied cudart_static dependency
## Performance Optimizations
  - Optimized nvcompBatchedDeflateGetDecompressSizeAsync. Now 2-3x faster on A100.

# nvcomp 2.3.2 (2022-06-24)
## Bug Fixes
  - Fixed various bugs in ZSTD decompression implementation
  - Fixed the issue of deflate compression could not be correctly decompressed by zlib::inflate().

# nvcomp 2.3.1 (2022-06-15)
## Bug Fixes
  - Fixed various bugs in ZSTD decompression implementation
  - Fixed various bugs in ANS compression implementation
  - Fix hang in GDeflate high-compression mode for large files
  - Fix bug in library build that required dynamic link to cudart.
## Interface Changes
  - Added new API, nvcompBatched\<Format\>DecompressGetTempSizeEx(). 
  This provides an optional capability for providing the total decompressed
  size to the API, which for some formats can dramatically reduce the required
  temp size.

# nvcomp 2.3.0 (2022-04-29)
## New Features
  - Support ZSTD decompression in the LLIF
  - Deflate support (RFC 1951)
  - Modified-CRC32 checksum support added to HLIF. Includes optional verification of HLIF-compressed buffers intended for error detection
## Bug fixes
  - Added Pascal GPU architecture support for all compressors
## Performance Optimizations
  - Performance optimizations in ANS compression / decompression, leading to ~100% speedup in compression and ~50% speedup in decompression
  - Developed algorithmic improvements to GDeflate's high-compression mode. This is now 30-40x faster on average while producing the same output as the previous version
## Infrastructure
  - Improvements to the benchmarking interface for LLIF -- common argument APIs
  
# nvcomp 2.2.0 (2022-02-07)
## New Features
 - Entropy-only mode for GDeflate
 - New high-level interface
 - Windows support
 - Support for GPU-accelerated ANS

## Interface Changes

### High level interface
 - High level interface is now standardized across compressor formats. 
 - This interface provides a single nvcompManagerBase object that
 can do compression and decompression. Users can now decompress nvcomp-compressed
 files without knowing how they were compressed. The interface also can manage scratch space 
 and splitting the input buffer into independent chunks for parallel processing.

### API Consolidation
 - nvCOMP now supports only the low-level batch API and the new high level interface

# nvcomp 2.1.0 (2021-10-28)

## New Features
 - New release of low-level batched API for Cascaded and Bitcomp methods.
 - New high-throughput and high-compression-ratio GPU compressors in GDeflate

## Interface Changes
 - Update batched/low-level compression interfaces to take an options parameter,
 to allow configuring future compression algorithms.
 - Update batched/low-level decompression interfaces to output the decompressed
 size (or 0 if an error occurs).
 - Add bounds checking to batched/low-level decompression routines, such that
 if an invalid compressed data stream is provided, 0 will be written for the
 output size, rather than generating an illegal memory access.
 - Fix LZ4 to support chunk sizes < 32 KB. 

## Performance Optimizations
 - Improve performance of Snappy compression by ~10% in some configurations.
 - Add an optimization to the LZ4 compressor based on specification of input data as
 char, short, or int, rather than just treating the input as raw bytes. 
 - Optimization to reduce the LZ hash table size when compressing smaller chunks.
 - Improved compression performance in GDeflate with the high-throughput option
 - Improved decompression performance in GDeflate (10-75% depending on the dataset)

## Bug Fixes
 - Fix LZ4 CPU compression example.
 - Fix temp allocation size bug in `benchmark_template_chunked`.

## Infrastructure
 - Update CMakeLists to compile nvcomp with -fPIC enabled.
 - Add a new script for benchmarking compression algorithms.
 - Add unit tests for the Snappy decompressor that tests decompression on legally
 formatted files that won't be generated by the nvcomp compressor due to 
 configuration.
 - Update CMakeLists to suppress warnings about missing nvcomp external dependencies
 when the user didn't indicate they wanted to include them.
 - Update CMakeLists to allow install into include folder that the user does not have 
 ownership of.
 
# nvcomp 2.0.2 (2021-06-30)

- Add example `lz4_cpu_decompression` to compress on the GPU with nvCOMP and
  decompress on the CPU with `liblz4`.
- Add CMake option for building a static library.
- Fix bug in LZ4 compression kernel to comply with LZ4 end of block
  restrictions.
- Fix temp allocation size bug in `benchmark_lz4_chunked`.

# nvcomp 2.0.1 (2021-06-08)

- Improve CMake setup for using nvCOMP as a submodule. This includes marking
dependencies as PRIVATE, and adding options for building examples, tests, and
benchmarks (e.g., `-DBUILD_EXAMPLES=ON`, `-DBUILD_TESTS=ON`, and
`-DBUILD_BENCHMARKS=ON`).
- Fix double free error in `benchmark_snappy_synth`.
- Fix copy direction in Cascaded compression when the output size on the GPU.
- Improve testing coverage.
- Mark the generic decompression interfaces defined in `include/nvcomp.h` as
deprecated.

# nvcomp 2.0.0 (2021-04-28)

- Replace previous C, and C++ APIs.
- Added Snappy compression (batched interface).
- Added support for using Bitcomp and GDeflate external compressors.
- Added `/examples` folder demonstrating use cases interface with CPU
implementations of LZ4 and GDeflate, as well as GPU Direct Storage.
- Improve support for Windows in benchmark implementations.
- Made usage of `std::uniform_int_distribution<>` in the benchmarks
conform to the C++14 standard.
- Fix issue in Cascaded compression when using the default configuration
('auto'), for small inputs.

# nvcomp 1.2.3 (2021-04-07)

- Fix bug in LZ4 compression kernel for the Pascal architecture.

# nvcomp 1.2.2 (2021-02-08)

- Fix linking errors in Clang++.
- Fix error being incorrectly returned by Cascaded compression when output memory was initialized to
all `-1`'s.
- Fix C++17 style static assert.
- Fix prematurely freeing memory in Cascaded compression.
- Fix input format and usage messaging for benchmarks.

# nvcomp 1.2.1 (2020-12-21)

- Fix compile error and unit tests for cascaded selector.

# nvcomp 1.2.0 (2020-12-19)

- Add the Cascaded Selector and Cascaded Auto set of interfaces for
automatically configuring cascaded compression.
- Generally improve error handling and messaging.
- Update CMake configuration to support CCache.

# nvcomp 1.1.1 (2020-12-02)

- Add all-gather benchmark.
- Add sm80 target if CUDA version is 11 or greater.

# nvcomp 1.1.0 (2020-10-05)

- Add batch C interface for LZ4, allowing compressing/decompressing multiple
inputs at once.
- Significantly improve performance of LZ4 compression.

# nvcomp 1.0.2 (2020-08-12)

- Fix metadata freeing for LZ4, to avoid possible mismatch of `new[]` and
`delete`.


# nvcomp 1.0.1 (2020-08-07)

- Fixed naming of nvcompLZ4CompressX functions in `include/lz4.h`, to have the
`nvcomp` prefix.
- Changed CascadedMetadata::Header struct initialization to work around
internal compiler error.


# nvcomp 1.0.0 (2020-07-31)

- Initial public release.

