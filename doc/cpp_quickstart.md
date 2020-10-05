# C++ Quick Start Guide

NVCOMP provides a C++ interface, which simplifies use of the library, by
throwing exceptions and managing state inside of objects. 

* [Compressing the Data](#compressing-the-data)
* [Transferring the Data (Optional)](#transferring-the-data-optional)
* [Decompressing the Data](#decompressing-the-data)

To use NVCOMP's C++ interface with the Cascaded compression scheme below,
you will need to include `nvcomp.hpp` and `cascaded.hpp`.

```C++
#include "cascaded.hpp"
#include "nvcomp.hpp"
```

## Compressing the Data

In order to compress data on the GPU, you must first create a 
`Compressor` object. In this case, we are using the `CascadedCompressor`, with
two Run Length Encoding layers, a delta layer, and bit-packing on the final
output. The type of our data is `int`.

```c++
nvcomp::CascadedCompressor<int> compressor(
    uncompressed_data, uncompressed_count, 2, 1, true);
```

Once this is done, we must get the required amount of temporary GPU space, to
perform the compression, and allocate it.

```c++
const size_t temp_size = compressor.get_temp_size();

void * temp_space;
cudaMalloc(&temp_space, temp_size);
```

Next, we need to get the required size of the output location, and allocate it.
The required size will often be larger than the actual size of compressed data
for the `CascadedCompressor`. This is because the exact size of the output is
not known until compression has run.

```c++
size_t output_size = compressor.get_max_output_size(
    temp_space, temp_size);
    
void * output_space;
cudaMalloc(&output_space, output_size);
```

Once we have our temporary and output memory allocations created, we can launch
the compression task.

```c++
nvcompError_t status;
compressor.compress_async(temp_space,
    temp_size, output_space, &output_size, stream);
```

In this case, `output_size` is in page-able memory, and as a
result, compression will be synchronous as it copies the value 
from device memory.

Pinned memory can be used for `output_size` to allow compression
to be performed asynchronously. However, the stream will need to be
synchronized on before attempt to read from `output_size`.

## Transferring the Data (Optional)

Once the data has been compressed, it can be transferred to the host, a file,
or other devices. The compression information is stored at the front of the
compressed data, so a simple `cudaMemcpy` can be used.

```c++
cudaMemcpy(host_buffer, output_space, output_size, cudaMemcpyDeviceToHost);

...

cudaMemcpy(compressed_data, host_buffer, output_size, cudaMemcpyHostToDevice);
```


## Decompressing the Data

When decompressing the data, we can use the generic `Decompressor` class, as it
will detect what the underlying compression used was. However, it is important
to match the template type, with that which was compressed.

```c++
nvcomp::Decompressor<int> decompressor(compressed_data, output_size, stream);
```

The decompressor will use the stream for copying compression metadata down from
the device, and synchronize afterwards.

We can then get the required amount of temporary GPU space in bytes, needed for
decompression, from this object, and allocate it.

```c++
const size_t temp_size = decompressor.get_temp_size();

void * temp_space;
cudaMalloc(&temp_space, temp_size);
```

Next, we need to get the number of elements that will be uncompressed, and
allocate space for it. The count will be exact.

```c++
const size_t output_count = decompressor.get_num_elements();
    
int * output_space;
cudaMalloc((void**)&output_space, output_count*sizeof(int));
```

Finally we can launch our decompression task on a stream.

```c++
nvcompError_t status;
decompressor.decompress_async(temp_space, temp_size, output_space, output_count, 
    stream);
```

The variable `output_size` here is passed by value rather than reference, so
decompression will be asynchronous without the need for pinned memory.