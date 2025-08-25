#pragma once

#include <stdlib.h>
#include <string>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>

static int warp_size = 0;
static int max_threads_per_block = 0;

#define CUDA_CHECK(x)                                                                                    \
  do {                                                                                                 \
    cudaError_t err = x;                                                                             \
    if (err != cudaSuccess) {                                                                        \
      fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
              cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
      abort();                                                                                     \
    }                                                                                                \
  } while (0)

#define CUDA_CHECK2(x, msg)                                                                                    \
  do {                                                                                                 \
    cudaError_t err = x;                                                                             \
    if (err != cudaSuccess) {                                                                        \
      fprintf(stderr, "[%s] CUDA error in %s at %s:%d: %s (%s=%d)\n", msg.c_str(), __FUNCTION__, __FILE__, __LINE__, \
              cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
      abort();                                                                                     \
    }                                                                                                \
  } while (0)

void* cuda_devicecopy(void* host, size_t size);

void* cuda_hostcopy(void* device, size_t size, std::string debug = "");

[[maybe_unused]] void* cuda_devicealloc(size_t size);

[[maybe_unused]] void* cuda_hostalloc(size_t size);

void* upload_cuda(void* host, size_t size);

void* download_cuda(void* device, size_t size, std::string debug);

void register_cuda_host(void* host, size_t size);

void free_cuda(void* device);

void unregister_cuda_host(void* host);

void set_cuda_device(int device);

void init_cuda_stream(cudaStream_t* stream);
#endif