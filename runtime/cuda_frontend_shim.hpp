#pragma once
#include <cstdlib>
#include "cuda_runtime.h"
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
struct uint3 { unsigned x, y, z; };
extern const __device__ uint3 threadIdx, blockIdx, blockDim, gridDim;
extern "C" cudaError_t cudaConfigureCall(dim3, dim3, unsigned = 0, void* = nullptr);
extern "C" cudaError_t cudaSetupArgument(const void*, unsigned long, unsigned long);
extern "C" cudaError_t cudaLaunch(const void*);
extern "C" cudaError_t cudaLaunchKernel(const void*, dim3, dim3, void**, unsigned long = 0, void* = nullptr);
