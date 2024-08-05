> [!WARNING]
> This page will be retired soon. With the release of [nvCOMP v4.0](https://developer.nvidia.com/nvcomp-download), the examples and benchmarks in this repository have been moved to the [CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/nvCOMP) GitHub repository. Documentation has been moved to [NVIDIA Documentation Hub](https://docs.nvidia.com/cuda/nvcomp/index.html). Binary packages for nvCOMP can still be downloaded from the [nvCOMP Developer Page](https://developer.nvidia.com/nvcomp-download).

# What is nvCOMP?

nvCOMP is a CUDA library that features generic compression interfaces to enable developers to use high-performance GPU compressors and decompressors in their applications.

Example benchmarking results and a brief description of each algorithm are available on the [nvCOMP Developer Page](https://developer.nvidia.com/nvcomp).

From version 2.3 onwards, the compression / decompression source code will not be released. 

## Known issues
* Cascaded, GDeflate, zStandard, Deflate, Gzip and Bitcomp decompressors can only operate on valid input data (data that was compressed using the same compressor). Other decompressors can sometimes detect errors in the compressed stream. 
* Cascaded, zStandard and Bitcomp batched decompression C APIs cannot currently accept nullptr for actual_decompressed_bytes or device_statuses values. Deflate and Gzip cannot accept nullptr for device_statuses values. 
* The Bitcomp low-level batched decompression function is not fully asynchronous.
* Gzip low-level interface only provides decompression.
* The device API only supports the ANS format
* Gdeflate API test and ANS device test is not working on H100 with CTK 12.x, will be excluded in the X86_64 build.

## Download
* You can download the appropriate built binary packages from the [nvCOMP Developer Page](https://developer.nvidia.com/nvcomp-download)
* Choose the package that corresponds to your CUDA toolkit version, operating system, and arch
* For example, on linux, the package includes
```
include/ 
  nvcomp/ #nvcomp API headers
  gdeflate/ #Gdeflate CPU library headers
lib/
  libnvcomp.so
  <Other nvcomp libraries that are used internally by nvcomp's APIs>
  libnvcomp_gdeflate_cpu.so # CPU library for gdeflate
  cmake/ <Package files to allow use through cmake>
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
gzip_gpu_decompression {-f <input_file>}
```

## Building CPU and GPU Examples, GPU Benchmarks provided on Github
To build only the examples, you'll need cmake >= 3.18 and an nvcomp artifact. Then, you can follow the following steps from the top-level of your clone of nvCOMP from Github
```
cmake .. -DCMAKE_PREFIX_PATH=<path_to_nvcomp_install>
```

The `path_to_nvcomp_install` is the directory where you extracted the nvcomp artifact.

To compile the benchmarks too, you can add `-DBUILD_BENCHMARKS=1`, but note this is only provided for an additional example of building against the artifacts. The benchmarks are already provided in the artifact `bin/` folder.

## Logging

To enable logging, set the `NVCOMP_LOG_LEVEL` environment variable to an integer:
* 0 for no logging
* 1 for only error messages
* 2 for error and warning messages
* 3 for errors, warnings, and information logged for every low-level interface API call
* 4 or 5 for debug information, not yet supported

By default, log messages will be written to a file named `nvcomp_yyyy-mm-dd_hh-mm.log`, with the date and time filled in.  If the `NVCOMP_LOG_FILE` environment variable is set to a valid file path, messages will be logged to that file.  Specifying `stdout` or `stderr` as the file will log to the console via the appropriate pipe, with color.
