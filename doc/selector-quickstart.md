## Cascaded Auto-selector

The nvcomp cascaded compressor has many configuration options that may significantly impact compression ratio
for a given dataset.
Finding the best configration for a given dataset can be time-consuming, so we provide an auto-selector
to make using cascaded compression easier. We provide both C and C++ interfaces for the selector and also provide
methods that automatically run the selector during compression, making the process much easier for users. Our Auto-selector 
uses sampling and is
much faster than manually testing each configuration.  The selector can be used directly to determine the best
configuration, or it can be run automatically during compression.  When running directly, the selector can
also estimate the compression ratio that will be achieved for the given input data.

### Running the Selector

We provide both C and C++ interfaces for the selector. 
See `tests/test_cascaded_selector.cpp` for working code using both the C and C++ interfaces of
the selector. Running the Cascaded Selector by itself requires configuring, allocating temp space, and executing.
Once it completes, you can use the provided `format_opts` to call the Cascaded compressor as usual. 
The example code below shows how to run the Selector with the C interface to obtain
the `nvcompCascadedFormatOpts` that can be used to call Cascaded compression. 

```c++
// Set up options for the selector.  This is a good default setting.
nvcompCascadedSelectorOpts selector_opts;
selector_opts.sample_size = 1024;
selector_opts.num_samples = 100;
selector_opts.seed = 1; // seed used for random sampling.

// Get size and allocate temp space needed to run selector.
size_t selector_temp_bytes;
nvcompCascadedSelectorConfigure(&selector_opts, getnvcompType<T>(), uncompressed_bytes, &selector_temp_bytes);
void* d_temp_selector;
cudaMalloc(&d_temp_selector, selector_temp_bytes);

// Run the Selector to get the Cascaded format opts and estimate compression ratio
nvcompCascadedFormatOpts opts;
double estimate_ratio;
nvcompCascadedSelectorRun(&selector_opts, getnvcompType<T>(), uncompressed_data, uncompressed_bytes, 
    d_temp_selector, selector_temp_bytes, &opts, &estimate_ratio, stream);

// Now run compression as normal using the format opts
```

The C++ interface is similar but uses a new class called `CascadedSelector` (details in `include/nvcomp/cascaded.hpp`).
Below is the same example code using the C++ interface:

```c++
// Set up options for the selector.  This is a good default setting.
nvcompCascadedSelectorOpts selector_opts;
selector_opts.sample_size = 1024;
selector_opts.num_samples = 100;
selector_opts.seed = 1;

CascadedSelector<int> selector(in_data, in_bytes, selector_opts);

// Allocate temp space to run selector
size_t temp_bytes = selector.get_temp_size();
void* d_temp;
cudaMalloc(d_temp, temp_bytes);

double estimate_ratio;
nvcompCascadedFormatOpts opts = selector.select_config(d_temp, temp_bytes, &estimate_ratio, stream);
```

### Automatically using the selector during compression

For ease of use, we also provide an interface to run Cascaded compression without ever specifying the
format or any details of the selector. These calls automatically run both the selector and compression,
letting the user avoid extra API calls and added code complexity, while still using the Selector
to achieve the best compression ratio.  One drawback of this approach is that the compression
call is no longer asynchronous.  That is, the call synchronizes on the CUDA stream that is passed
into the API call.  

With the C interface, you can have the selector automatically run by passing NULL in place of the `format_opts`
parameter.  An example of this is below:

```c++
// Get size and allocate storage to run selector and perform compression
size_t temp_bytes;
nvcompCascadedCompressConfigure(NULL, getnvcompType<int>(), in_bytes, 
    &metadata_bytes, &temp_bytes, &out_bytes);

void* d_temp;
cudaMalloc(&d_temp, temp_bytes);
void* d_out;
cudaMalloc(&d_out, out_bytes);

// Run both the selector and compression, putting the compressed output in d_out
nvcompCascadedCompressAsync(NULL, getnvcompType<T>(), in_data, in_bytes, 
    d_temp, temp_bytes, d_out, out_bytes, stream);
```

As mentioned above, using the C++ interface to auto-run the selector during compression is very simple. You can use all 
of the CascadedCompressor methods as normal (detailed in the [C++ Quick Start Guide](cpp_quickstart.md)), and just 
use a constructor that does not take any cascaded format options as input:

```c++
CascadedCompressor compressor(getnvcompType<int>());
```

The compressor can then be used to `configure()`, and `compress_async()`.  The only change is that
the call to `compress_async` will automatically call the selector, causing it to synchronize on the CUDA stream.
No changes to decompression code are required when using the selector during compression like this.  
