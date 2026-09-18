#ifndef CUDA2OMP_CUDA_RUNTIME_H
#define CUDA2OMP_CUDA_RUNTIME_H

/*
 * Minimal CUDA Runtime API compatibility for CPU-only programs.
 *
 * All launches and memory operations in cuda2omp execute synchronously.  A
 * stream parameter, where supported by the translator, currently provides
 * sequential ordering semantics only; it does not introduce asynchronous
 * execution or overlap.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorInvalidDevice = 10,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorUnknown = 999
} cudaError_t;

typedef enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
} cudaMemcpyKind;

#ifdef __cplusplus
struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1)
      : x(x_), y(y_), z(z_) {}
};
#endif

#if defined(__cplusplus)
#define CUDA2OMP_THREAD_LOCAL thread_local
#else
#define CUDA2OMP_THREAD_LOCAL _Thread_local
#endif

#ifdef __cplusplus
extern "C" {
#endif

static CUDA2OMP_THREAD_LOCAL cudaError_t cuda2omp_last_error = cudaSuccess;

static inline cudaError_t cuda2omp_record_error(cudaError_t error) {
  cuda2omp_last_error = error;
  return error;
}

static inline cudaError_t cudaMalloc(void **pointer, size_t bytes) {
  if (pointer == NULL || bytes > (size_t)PTRDIFF_MAX)
    return cuda2omp_record_error(cudaErrorInvalidValue);
  if (bytes == 0) {
    *pointer = NULL;
    return cuda2omp_record_error(cudaSuccess);
  }
  *pointer = malloc(bytes);
  return cuda2omp_record_error(*pointer != NULL ? cudaSuccess
                                                : cudaErrorMemoryAllocation);
}

static inline cudaError_t cudaFree(void *pointer) {
  free(pointer);
  return cuda2omp_record_error(cudaSuccess);
}

static inline cudaError_t cudaMemcpy(void *destination, const void *source,
                                     size_t bytes, cudaMemcpyKind kind) {
  if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault)
    return cuda2omp_record_error(cudaErrorInvalidMemcpyDirection);
  if (bytes > (size_t)PTRDIFF_MAX ||
      (bytes != 0 && (destination == NULL || source == NULL)))
    return cuda2omp_record_error(cudaErrorInvalidValue);
  if (bytes != 0)
    memmove(destination, source, bytes);
  return cuda2omp_record_error(cudaSuccess);
}

static inline cudaError_t cudaMemset(void *destination, int value,
                                     size_t bytes) {
  if (bytes > (size_t)PTRDIFF_MAX || (bytes != 0 && destination == NULL))
    return cuda2omp_record_error(cudaErrorInvalidValue);
  if (bytes != 0)
    memset(destination, value, bytes);
  return cuda2omp_record_error(cudaSuccess);
}

static inline cudaError_t cudaDeviceSynchronize(void) {
  return cuda2omp_record_error(cudaSuccess);
}

static inline cudaError_t cudaPeekAtLastError(void) {
  return cuda2omp_last_error;
}

static inline cudaError_t cudaGetLastError(void) {
  cudaError_t error = cuda2omp_last_error;
  cuda2omp_last_error = cudaSuccess;
  return error;
}

static inline const char *cudaGetErrorString(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorInitializationError: return "initialization error";
    case cudaErrorInvalidDevice: return "invalid device ordinal";
    case cudaErrorInvalidMemcpyDirection: return "invalid memcpy direction";
    case cudaErrorUnknown: return "unknown error";
    default: return "unrecognized error code";
  }
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef CUDA2OMP_THREAD_LOCAL
#endif
