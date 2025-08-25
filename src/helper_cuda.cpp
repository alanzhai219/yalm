#include <stdlib.h>
#include <stdio.h>
#include "helper_cuda.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>

void* cuda_devicecopy(void* host, size_t size) {
  void* device = NULL;
  CUDA_CHECK(cudaMalloc(&device, size));
  CUDA_CHECK(cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice));
  return device;
}

void* cuda_hostcopy(void* device, size_t size, std::string debug) {
  void* host = NULL;
  CUDA_CHECK2(cudaMallocHost(&host, size), debug);
  CUDA_CHECK2(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost), debug);
  return host;
}

[[maybe_unused]] void* cuda_devicealloc(size_t size) {
  void* ptr = NULL;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

[[maybe_unused]] void* cuda_hostalloc(size_t size) {
  void* ptr = NULL;
  CUDA_CHECK(cudaHostAlloc(&ptr, size, 0));
  return ptr;
}

void* upload_cuda(void* host, size_t size) {
  return cuda_devicecopy(host, size);
}

void* download_cuda(void* device, size_t size, std::string debug) {
  return cuda_hostcopy(device, size, debug);
}

void register_cuda_host(void* host, size_t size) {
  CUDA_CHECK(cudaHostRegister(host, size, cudaHostRegisterDefault));
}

void free_cuda(void* device) {
  CUDA_CHECK(cudaFree(device));
}

void unregister_cuda_host(void* host) {
  CUDA_CHECK(cudaHostUnregister(host));
}

void set_cuda_device(int device) {
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device));
}

void init_cuda_stream(cudaStream_t* stream) {
  CUDA_CHECK(cudaStreamCreate(stream));
}
#endif