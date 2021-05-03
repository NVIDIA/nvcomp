# Low-level C Quick Start Guide

Some applications require compressing or decompressing multiple small inputs,
so we provide an additional API to do this efficiently. These API calls combine
all compression/decompression
into a single execution, greatly improving performance compared with running
each input individually.  Note that currently, this is only available with the
C API and using the LZ4 and Snappy compressors. This API relies on the user to
split the data into chunks, as well as manage metadata information such as
compressed and uncompressed chunk sizes. When splitting data, for best
performance, chunks should be relatively equal size to achieve good
load-balancing as well as extract sufficient parallelism. So in the case that
there are multiple inputs to compress, it may still be best to break each one
up into smaller chunks.

In this guide, we walk through how to use this API with the LZ4 compressor,
making use of the functions declared in `include/nvcomp/lz4.h`.

## Batched Compression

The batched LZ4 compression C API call takes, as input, a array of input
pointers on the GPU (`d_comp_input`), their uncompressed sizes
(`d_uncomp_sizes`), and as output writes to a set of output locations on the
GPU (`d_comp_output`), and compressed sizes (`d_comp_sizes`).

So first, we need to split the data into chunks.
The below code snippet, assumes we have a single array of data to compress
on the GPU (`d_in_data`, of size `in_bytes`).

```c++
// compute chunk sizes 
const size_t chunk_size = 65536;
const size_t num_chunks = (in_bytes + chunk_size - 1) / chunk_size;
size_t * chunk_sizes;
uncomp_sizes = cudaMallocHost((void**)&uncomp_sizes,
    sizeof(*uncomp_sizes)*num_chunks);
for (size_t i = 0; i < num_chunks; ++i) {
  if (i + 1 < num_chunks) {
    uncomp_sizes[i] = chunk_size;
  } else {
    // last chunk may be smaller
    uncomp_sizes[i] = in_bytes - (chunk_size*i);
  }
}

// setup input pointers
void ** comp_input;
cudaMallocHost((void**)&comp_input, sizeof(*comp_input)*num_chunks);
for (size_t i = 0; i < num_chunks; ++i) {
  comp_input[i] = ((char*)d_in_data) + (chunk_size*i);
}
```

Next, we need to copy this input information to the GPU.
```c++
// copy chunk sizes to the GPU
size_t * d_uncomp_sizes;
cudaMalloc((void**)&d_uncomp_sizes, sizeof(*d_uncomp_sizes));
cudaMemcpyAsync(d_uncmop_sizes, uncomp_sizes,
    sizeof(*d_uncomp_sizes)*num_chunks, cudaMemcpyHostToDevice, stream);

// copy input pointers to the GPU
void ** d_comp_input;
cudaMalloc((void**)&d_comp_input, sizeof(*d_comp_input)*num_chunks);
cudaMemcpyAsync(d_comp_input, comp_input, sizeof(*d_comp_input)*num_chunks,
    cudaMemcpyHostToDevice, stream);
```

We also need to allocate the temporary workspace and output space needed by the
compressor.

```c++
// setup temporary space
size_t temp_bytes;
nvcompBatchedLZ4CompressGetTempSize(num_chunks, chunk_size, &temp_bytes);
void* d_comp_temp;
cudaMalloc(&d_comp_temp, temp_bytes);

// get the maxmimum output size for each chunk
size_t max_out_bytes;
nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, &max_out_bytes);

// allocate output space on the device
void** comp_output;
cudaMallocHost((void**)&comp_output, sizeof(*comp_output)*num_chunks);
for(size_t i=0; i<num_chunks; ++i) {
    cudaMalloc(&comp_output[i], max_out_bytes);
}

// setup pointers to the output space on the device
void** d_comp_output;
cudaMalloc((void**)&d_comp_output, sizeof(*d_comp_output)*num_chunks);
cudaMemcpyAsync(d_comp_output, comp_output, sizeof(*d_comp_output)*num_chunks,
    cudaMemcpyHostToDevice, stream);

// allocate space for compressed chunk sizes to be written to
size_t * d_comp_sizes;
cudaMalloc((void**)&d_comp_sizes, sizeof(*d_comp_sizes)*num_chunks);
```

An alternative to setting up all of this information using the CPU, and then
copying it to the GPU, would be to use a custom kernel to do this work directly
on the GPU.

Once we have everything setup, we can launch compression asynchronously.

```c++
nvcompBatchedLZ4CompressAsync(d_comp_input, d_uncomp_sizes, chunk_size,
    num_chunks, d_comp_temp, temp_bytes, d_comp_output, d_comp_sizes, stream);
```

This call compresses each input `d_comp_input[i]`,
placing the compressed output in the corresponding output list,
`d_comp_output[i]`, and its compressed size in `d_comp_sizes`.

## Batched Decompression

Decompression can be similarly performed on a batch of multiple compressed
input lists. As no metadata is stored with the compressed data, chunks can be
re-arranged as well decompressed with other originally not compressed in the
same batch.

In addition to providing an array on the GPU of pointers on the compressed chunks
(`d_comp_output`),
the number of inputs (`num_chunks`), and their sizes in
bytes (`d_comp_sizes`) to the decompressor, we also need the maximum
uncompressed chunk size (`chunk_size`).

Similarly, to compression, we first need to allocate temporary workspace.

```c++
size_t temp_bytes;
nvcompBatchedLZ4DecompressGetTempSize(num_chunks, chunk_size, &temp_bytes);
void * d_decomp_temp;
cudaMalloc(&d_decomp_temp, temp_bytes);
```

Next, output space must be allocated, using information about the uncompressed
chunk sizes. Rather than recompute the uncompressed chunk sizes, we will just
re-use the ones we calculated during compression.

```c++
// allocate the output space using the original data size
void * d_out_data;
cudaMalloc(&d_out_data, in_bytes);

// setup output pointers
void ** decomp_output;
cudaMallocHost((void**)&decomp_output, sizeof(*decomp_output)*num_chunks);
for (size_t i = 0; i < num_chunks; ++i) {
  decomp_output[i] = ((char*)d_out_data) + (chunk_size*i);
}

// copy output pointers to the GPU
void ** d_decomp_output;
cudaMalloc(&d_decomp_output, sizeof(*d_decomp_output)*num_chunks);
cudaMemcpyAsync(d_decomp_output, decomp_output,
    sizeof(*d_decomp_output)*num_chunks, cudaMemcpyHostToDevice, stream);
```

Asynchronous decompression can then be launched.

```c++
nvcompBatchedLZ4DecompressAsync(
    d_comp_output, d_comp_sizes, d_uncomp_sizes, chunk_size, num_chunks,
    d_decomp_temp, temp_bytes, d_decomp_output, stream);
```

This decompresses each input, `d_comp_output[i]`, and places the decompressed
result in the corresponding output list, `d_decomp_output[i]`.
