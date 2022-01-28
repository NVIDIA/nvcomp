/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "nvcomp/cascaded.hpp"
#include "benchmark_hlif.hpp"

#include <string.h>

using namespace nvcomp;

static void print_usage()
{
  printf("Usage: benchmark_lz4 [OPTIONS]\n");
  printf("  %-35s Binary dataset filename (required).\n", "-f, --filename");
  printf("  %-35s GPU device number (default 0)\n", "-g, --gpu");
  printf("  %-35s Data type (default 'char', options are 'char', 'short', 'int')\n", "-t, --type");
  printf(
      "  %-35s Output GPU memory allocation sizes (default off)\n",
      "-m --memory");
  exit(1);
}


int main(int argc, char* argv[])
{
  char* fname = NULL;
  int gpu_num = 0;
  int verbose_memory = 0;
  nvcompType_t data_type = nvcompBatchedCascadedDefaultOpts.type;
  int num_rles = nvcompBatchedCascadedDefaultOpts.num_RLEs;
  int num_deltas = nvcompBatchedCascadedDefaultOpts.num_deltas;
  int num_bps = nvcompBatchedCascadedDefaultOpts.use_bp;

  // Parse command-line arguments
  char** argv_end = argv + argc;
  argv += 1;
  while (argv != argv_end) {
    char* arg = *argv++;
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-?") == 0) {
      print_usage();
      return 1;
    }
    if (strcmp(arg, "--memory") == 0 || strcmp(arg, "-m") == 0) {
      verbose_memory = 1;
      continue;
    }

    // all arguments below require at least a second value in argv
    if (argv >= argv_end) {
      print_usage();
      return 1;
    }

    char* optarg = *argv++;
    if (strcmp(arg, "--filename") == 0 || strcmp(arg, "-f") == 0) {
      fname = optarg;
      continue;
    }
    if (strcmp(arg, "--gpu") == 0 || strcmp(arg, "-g") == 0) {
      gpu_num = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--type") == 0 || strcmp(arg, "-t") == 0) {
      if (strcmp(optarg, "char") == 0) {
        data_type = NVCOMP_TYPE_CHAR;
      } else if (strcmp(optarg, "short") == 0) {
        data_type = NVCOMP_TYPE_SHORT;
      } else if (strcmp(optarg, "int") == 0) {
        data_type = NVCOMP_TYPE_INT;
      } else if (strcmp(optarg, "longlong") == 0) {
        data_type = NVCOMP_TYPE_LONGLONG;
      } else {
        print_usage();
        return 1;
      }
      continue;
    }
    if (strcmp(arg, "--num_rles") == 0 || strcmp(arg, "-r") == 0) {
      num_rles = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--num_deltas") == 0 || strcmp(arg, "-d") == 0) {
      num_deltas = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--num_bps") == 0 || strcmp(arg, "-b") == 0) {
      num_bps = atoi(optarg);
      continue;
    }

    print_usage();
    return 1;
  }

  if (fname == NULL) {
    print_usage();
    return 1;
  }

  cudaSetDevice(gpu_num);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompBatchedCascadedOpts_t options = nvcompBatchedCascadedDefaultOpts;
  options.type = data_type;
  options.num_RLEs = num_rles;
  options.num_deltas = num_deltas;
  options.use_bp = (num_bps != 0);
  CascadedManager batch_manager{options, stream};

  run_benchmark_from_file(fname, batch_manager, verbose_memory, stream);
  CudaUtils::check(cudaStreamDestroy(stream));

  return 0;
}
