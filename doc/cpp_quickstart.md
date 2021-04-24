# C++ Quick Start Guide

NVCOMP provides a C++ interface, which simplifies use of the library, by
throwing exceptions and managing state inside of objects. 

* [Compressing the Data](#compressing-the-data)
* [Transferring the Data (Optional)](#transferring-the-data-optional)
* [Decompressing the Data](#decompressing-the-data)

To use NVCOMP's C++ interface, you will need to include `nvcomp.hpp`
and the headers of the specific compressor you will be using.  For
the Cascaded compression scheme shown in examples below, this is in
`nvcomp/cascaded.hpp`. 

```C++
#include "nvcomp/cascaded.hpp"
#include "nvcomp.hpp"
```

## Compressing the Data

In order to compress data on the GPU, you must first create a 
`Compressor` object. In this case, we are using the `CascadedCompressor`, using
`int` as the input datatype, and compressing with 
two Run Length Encoding layers, a delta layer, and bit-packing on the final
output. Any data compressed by this compressor
will be compresssed with this format.
If the last 3 arguments are omitted from the Cascaded compressor, the format selector will be run 
to determine them automatically.  For more information see the [Cascaded Format Selector Guide](selector-quickstart.md).

```c++
nvcomp::CascadedCompressor<int> compressor(getnvcompType<int>(), 2, 1, true);
```

Once this is done, we can get the required amount of temporary GPU space to perform the compression,
and an esimate of the compressed output size.  These have to be allocated in order to perform the compression.
Note that the output size will often be larger than the actual size of the compressed data.  This
is because the exact size of the output is not known until compression has completed.

```c++
compressor.configure(uncompressed_bytes, &temp_bytes, &output_bytes);

void * temp_space;
cudaMalloc(&temp_space, temp_bytes);

void * output_space;
cudaMalloc(&output_space, output_bytes);
```

Once we have our temporary and output memory allocations created, we can launch
the compression task.  We must ensure that the `output_bytes` variable is directly accessible
by the GPU during compression.  So, we first define a pinned memory variable and copy the size to
it.  Note that this could also be simple GPU memory, but using a simple pinned variable allows
easy access by host code as well.

```c++
int * d_compressed_bytes;
cudaMallocHost(&d_compressed_bytes, sizeof(int));
*d_compressed_bytes = output_bytes;

compressor.compress_async(uncompressed_data, uncompressed_bytes,
    temp_space, temp_bytes, output_space, d_compressed_bytes, stream);
```

This will issue the compression kernel.  The final compressed output size
will bet written to `d_compressed_bytes` once it completes.  Since it runs
asynchronously, you will need to synchronize on the stream to
access the final output size.

## Transferring the Data (Optional)

Once the data has been compressed, it can be transferred to the host, a file,
or other devices. The compression information is stored at the front of the
compressed data, so a simple `cudaMemcpy` can be used.

```c++
cudaMemcpy(host_buffer, output_space, *d_compressed_bytes, cudaMemcpyDeviceToHost);

...

cudaMemcpy(compressed_data, host_buffer, *d_compressed_bytes, cudaMemcpyHostToDevice);
```


## Decompressing the Data

When decompressing the data, we can use the generic `Decompressor` class, as it
will detect what the underlying compression used was. However, it is important
to match the template type, with that which was compressed.

To decompress the data, we use the corresponding Decompressor object.  Since
all of the information about the data is included with the compressed data,
the Decompressor does not need any format or type information.

```c++
nvcomp::CascadedDecompressor decompressor();
```
To prepare the decompressor to work on a specific compressed input, it has
to be configured.  The configure operation computes the required temporary storage
needed to perform decompression, as well as the final decompressed output size (exact size).  Once
complete, we can allocate the temporary and output spaces.

Note that this operation extracts metadata from the compressed input that is on
the GPU and synchronizes on the input CUDA stream.

```c++
decompressor.configure(compressed_data, compressed_bytes, &temp_bytes, &uncompressed_bytes, stream);

void * temp_space;
cudaMalloc(&temp_space, temp_bytes);

void * uncompressed_output;
cudaMalloc(&uncompressed_output, uncompressed_bytes);
```

Finally we can launch our decompression task.

```c++
nvcompError_t status;
decompressor.decompress_async(compressed_data, compressed_bytes, temp_space, 
    temp_bytes, uncompressed_output, uncompressed_bytes, stream);
```

The variable `uncompressed_bytes` here is passed by value rather than reference, so
there is no need for GPU-accessible memory for the output size.
