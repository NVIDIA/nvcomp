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
