# What is nvCOMP?

nvCOMP is a CUDA library that features generic compression interfaces to enable developers to use high-performance GPU compressors and decompressors in their applications.

Example benchmarking results and a brief description of each algorithm are available on the [nvCOMP Developer Page](https://developer.nvidia.com/nvcomp).

## Version 2.4 Release

This minor release of nvCOMP completes support for zStandard compression (https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md) 

Beginning with this release, we provide Linux SBSA binaries.

This release also includes the following performance improvements:
  - GDeflate high-compression mode up to 2x faster.
  - ZSTD decompression up to 1.2x faster.
  - Deflate decompression up to 1.5x faster.
  - ANS compression improvements based on strong scaling allows for up to 7x higher compression and decompression throughput for files on the order of a few MB in size. Decompression throughput is improved by at least 20% on all tested files.

From version 2.3 onwards, the compression / decompression source code will not be released. We'll continue to maintain this Github for documentation and code sample purposes.

## Known issues
* Cascaded, GDeflate, zStandard, Deflate and Bitcomp decompressors can only operate on valid input data (data that was compressed using the same compressor). Other decompressors can sometimes detect errors in the compressed stream. 
* Cascaded zStandard, Deflate and Bitcomp batched decompression C APIs cannot currently accept nullptr for actual_decompressed_bytes or device_statuses values.
* The Bitcomp low-level batched decompression function is not fully asynchronous.
* HLIF is not available for Deflate or zStandard
* The Deflate batched decompression API doesn't currently support uncompressed data chunk sizes larger than the standard deflate block size (64 KB).

## Download
* You can download the appropriate built binary packages from the [nvCOMP Developer Page](https://developer.nvidia.com/nvcomp)
* For linux, choose the package that corresponds to your CUDA toolkit version
* For Windows, CUDA toolkit 11.7 is required
* On linux, the package includes
```
include/ 
  nvcomp/ #nvcomp API headers
  gdeflate/ #Gdeflate CPU library headers
lib/
  libnvcomp.so
  <Other nvcomp libraries that are used internally by nvcomp's APIs>
  libnvcomp_gdeflate_cpu.so # CPU library for gdeflate
bin/ 
  <benchmark scripts>
```

## Requirements
* Pascal (sm60) or higher GPU architecture is required. Volta (sm70)+ GPU architecture is recommended for best results. 
* To compile using nvCOMP as a dependency, you need a compiler with C++ 11 support (e.g. GCC 4.4, Clang 3.3, MSVC 2017).
  * To use the packages provided for Linux, you'll need to compile translation units that will be linked against nvcomp using the old ABI, i.e. for GCC `-D_GLIBCXX_USE_CXX11_ABI=0`

## nvCOMP library API Descriptions

Please view the following guides for information on how to use the two APIs provided by the library. Each of these guides links to a compilable example for further reference. 
* [High-level Quick Start Guide](doc/highlevel_cpp_quickstart.md)
* [Low-level Quick Start Guide](doc/lowlevel_c_quickstart.md)

## GPU Benchmarking

GPU Benchmark source are included in the binary releases. Source code for the benchmarks is also provided here on Github to provide additional examples on how to use nvCOMP. For further information on how to execute the benchmarks, please view [Benchmarks Page](doc/Benchmarks.md)

## CPU compression examples

We provide some examples of how you might use CPU compression and GPU decompression or vice versa for LZ4 GDeflate and Deflate. These require some external dependencies, namely:
- [zlib](https://github.com/madler/zlib) for the GDeflate and Deflate CPU compression/decompression example (`zlib1g-dev` on debian based systems)
- [LZ4](https://github.com/lz4/lz4) for the LZ4 CPU compression example (`liblz4-dev` on debian based systems)
- [libdeflate](https://github.com/ebiggers/libdeflate) for the Deflate CPU compression/decompression example (`libdeflate-dev` on debian based systems)

The CPU example executables are:
```
gdeflate_cpu_compression {-f <input_file>}
lz4_cpu_compression {-f <input_file>}
lz4_cpu_decompression {-f <input_file>}
deflate_cpu_compression {-a <0 libdeflate, 1 zlib_compress2, 2 zlib_deflate> -f <input_file>}
deflate_cpu_decompression {-a <0 libdeflate, 1 zlib_inflate> -f <input_file>}
```

## Building CPU and GPU Examples, GPU Benchmarks provided on Github
To build only the examples, you'll need cmake >= 3.18 and an nvcomp artifact. Then, you can follow the following steps from the top-level of your clone of nvCOMP from Github
```
cmake .. -DCMAKE_PREFIX_PATH=<path_to_nvcomp_install>
```

The `path_to_nvcomp_install` is the directory where you extracted the nvcomp artifact.

To compile the benchmarks too, you can add `-DBUILD_BENCHMARKS=1`, but note this is only provided for an additional example of building against the artifacts. The benchmarks are already provided in the artifact `bin/` folder.

