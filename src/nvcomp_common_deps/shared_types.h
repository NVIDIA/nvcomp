#pragma once


typedef enum nvcompStatus_t
{
  nvcompSuccess = 0,
  nvcompErrorInvalidValue = 10,
  nvcompErrorNotSupported = 11,
  nvcompErrorCannotDecompress = 12,
  nvcompErrorCudaError = 1000,
  nvcompErrorInternal = 10000,
} nvcompStatus_t;
