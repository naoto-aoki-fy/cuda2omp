#pragma once
#include <cstdlib>
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
struct uint3 { unsigned x, y, z; };
struct dim3 { unsigned x, y, z; constexpr dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){} };
extern const __device__ uint3 threadIdx, blockIdx, blockDim, gridDim;
extern "C" int cudaConfigureCall(dim3, dim3, unsigned = 0, void* = nullptr);
extern "C" int cudaSetupArgument(const void*, unsigned long, unsigned long);
extern "C" int cudaLaunch(const void*);
