# What is nvCOMP?

nvCOMP is a CUDA library that features generic compression interfaces to enable developers to use high-performance GPU compressors and decompressors in their applications.

nvCOMP 2.0.0 includes Cascaded, LZ4, and Snappy compression methods.
It also adds support for the external Bitcomp and GDeflate methods.
Cascaded compression methods demonstrate high performance with up to 500 GB/s
throughput and a high compression ratio of up to 80x on numerical data from
analytical workloads.
Snappy and LZ4 methods can achieve up to 100 GB/s compression and decompression
throughput depending on the dataset, and show good compression ratios for
arbitrary byte streams.

Below are compression ratio and performance plots for three methods available in nvCOMP (Cascaded, Snappy and LZ4). Each column shows results for a single column from an analytical dataset derived from [Fannie Mae’s Single-Family Loan Performance Data](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html). The numbers were collected on a NVIDIA A100 80GB GPU (with ECC on). 

![compression ratio](/doc/compression_ratio.svg)

![compression performance](/doc/compression_performance_a100.svg)

![decompression performance](/doc/decompression_performance_a100.svg)

nvCOMP 2.0.0 features new flexible APIs:
* [**Low-level**](doc/lowlevel_c_quickstart.md) is targeting advanced users —
  metadata and chunking must be handled outside of nvCOMP, low-level nvCOMP
  APIs perform batch compression/decompression of multiple streams, they are
  light-weight and fully asynchronous.
* [**High-level**](doc/highlevel_cpp_quickstart.md) is provided for ease of use —
  metadata and chunking is handled internally by nvCOMP, this enables the
  easiest way to ramp up and use nvCOMP in applications, some of the high-level
  APIs are synchronous and for best performance/flexibility it’s recommended to
  use the low-level APIs.

Please note, that in nvCOMP 2.0.0 some compressor are only available either through the Low-level API or through the High-level API.

Below you can find instructions on how to build the library, reproduce our benchmarking results, a guide on how to integrate into your application and a detailed description of the compression methods. Enjoy!

# Version 2.0.0 Release

This release of nvCOMP introduces new interfaces and compression methods.

## Known issues

* Cascaded compression requires a large amount of temporary workspace to
operate. Current workaround is to compress/decompress large datasets in pieces,
re-using temporary workspace for each piece.

# Requirements
Pascal (sm60) or higher GPU architecture is required. Volta+ GPU architecture is recommended for best results.

# Building the library, with nvCOMP extensions
To configure nvCOMP extensions, simply define the `NVCOMP_EXTS_ROOT` variable
to allow CMake to find the library

First, download nvCOMP extensions from the [nvCOMP Developer Page](https://developer.nvidia.com/nvcomp).
There two available extensions.
1. Bitcomp
2. GDeflate
```
git clone https://github.com/NVIDIA/nvcomp.git
cd nvcomp
mkdir build
cd build
cmake -DNVCOMP_EXTS_ROOT=/path/to/nvcomp_exts/${CUDA_VERSION} ..
make -j4
```

# Building the library, without nvCOMP extensions
nvCOMP uses CMake for building. Generally, it is best to do an out of source build:
```
git clone https://github.com/NVIDIA/nvcomp.git
mkdir build
cd build
cmake ..
make -j
```

If you're building using CUDA 10 or less, you will need to specify a path to
[CUB](https://github.com/thrust/cub) on your system, of at least version
1.9.10.

```
cmake -DCUB_DIR=<path to cub repository>
```

# How to use the library in your code

* [High-level Quick Start Guide](doc/highlevel_cpp_quickstart.md)
* [Low-level Quick Start Guide](doc/lowlevel_c_quickstart.md)
* [Cascaded Format Selector Guide](doc/selector-quickstart.md)

# Further information about our compression algorithms

* [Algorithms overview](doc/algorithms_overview.md)

# Running benchmarks
To obtain TPC-H data:
- Clone and compile https://github.com/electrum/tpch-dbgen
- Run `./dbgen -s <scale factor>`, then grab `lineitem.tbl`

To obtain Mortgage data:
- Download any of the archives from https://rapidsai.github.io/demos/datasets/mortgage-data
- Unpack and grab `perf/Perforamnce_<year><quarter>.txt`, e.g. `Perforamnce_2000Q4.txt`

Convert CSV files to binary files:
- `benchmarks/text_to_binary.py` is provided to read a `.csv` or text file and output a chosen column of data into a binary file
- For example, run `python benchmarks/text_to_binary.py lineitem.tbl <column number> <datatype> column_data.bin '|'` to generate the binary dataset `column_data.bin` for TPC-H lineitem column `<column number>` using `<datatype>` as the type
- *Note*: make sure that the delimiter is set correctly, default is `,`

Run tests:
- Run `./bin/benchmark_cascaded_auto` or `./bin/benchmark_lz4` with `-f column_data.bin <options>` to measure throughput.

Below are some example benchmark results on a RTX 3090 for the Mortgage 2000Q4 column 0:

```
$ ./bin/benchmark_cascaded_auto -f ../../nvcomp-data/perf/2000Q4.bin -t long
----------
uncompressed (B): 81289736
comp_size: 2047064, compressed ratio: 39.71
compression throughput (GB/s): 225.60
decompression throughput (GB/s): 374.95
```

```
$ ./bin/benchmark_lz4 -f ../../nvcomp-data/perf/2000Q4.bin
----------
uncompressed (B): 81289736
comp_size: 3831058, compressed ratio: 21.22
compression throughput (GB/s): 36.64
decompression throughput (GB/s): 118.47
```
