## Batched Execution on Multiple Inputs

Some applications require compressing or decompressing multiple small inputs, so we provide
an additional API to do this efficiently with the LZ4 compressor/decompressor.  These API
calls combine all compression/decompression into a single execution, greatly improving
performance compared with running each input individually.  Note that currently,
this is only available with the C API and using the LZ4 compressor.

### Batched Compression

The batched LZ4 compression C API call takes, as input, a array of input pointers (`in_data`), 
the number of input lists (`num_inputs`), and their sizes in bytes (`input_byes`).  As with individual
list compression, you must first get temporary memory and output sizes and allocate the GPU memory.

```c++
size_t temp_bytes;
nvcompBatchedLZ4CompressGetTempSize(in_data, input_bytes, num_inputs, &comp_opts, &temp_bytes);
void* comp_temp;
cudaMalloc(&comp_temp, temp_bytes);

size_t out_bytes[num_inputs];
nvcompBatchedLZ4CompressGetOutputSize(in_data, input_bytes, num_inputs, &comp_opts, comp_temp, temp_bytes, out_bytes);
void* comp_output[num_inputs];
for(size_t i=0; i<num_inputs; ++i) {
    cudaMalloc(&comp_output[i], out_bytes[i]);
}
```

Once the temp buffer and each output buffer is allocated, a single asynchronous compression call can be made.

```c++
nvcompBatchedLZ4CompressAsync(in_data, input_bytes, num_inputs, &comp_opts, comp_temp, temp_bytes, comp_output, out_bytes, stream);
```

This call compresses each input `in_data[i]`, placing the compressed output in the corresponding output list, `comp_output[i]`.

### Batched Decompression

Decompression can be similarly performed on a batch of multiple compressed input lists.  Given a list of compressed inputs
(`comp_inputs`), the number of inputs (`num_inputs`), and their sizes in bytes (`comp_bytes`), you can perform the batched decompression 
by first extracting the necessary metadata.

```c++
void* metadata_ptr;
nvcompBatchedLZ4DecompressGetMetadata(comp_inputs, comp_bytes, num_inputs, &metadata_ptr, stream);
```

You can then get temp and output sizes (in bytes) and allocate the necessary memory.

```c++
size_t temp_bytes;
nvcompBatchedLZ4DecompressGetTempSize(metadata_ptr, &temp_bytes);
void* temp_ptr;
cudaMalloc(&temp_ptr, temp_bytes);

size_t decomp_bytes[num_inputs];
nvcompBatchedLZ4DecompressGetOutputSize(metadata_ptr, num_inputs, decomp_bytes);
void* decomp_output[num_inputs];
for(size_t i=0; i<num_inputs, ++i) {
    cudaMalloc(&decomp_output[i], decomp_bytes[i]);)
}
```

The entire batch can then be decompressed with a single asynchronous call.

```c++
nvcompBatchedLZ4DecompressAsync(comp_inputs, comp_bytes, num_inputs, temp_ptr, temp_bytes, metadata_ptr, decomp_output, decomp_bytes, stream);
```

This decompresses each input, `comp_inputs[i]`, and places the decompressed result in the corresponding output list, `decomp_output[i]`.