# cuda2omp prototype

`cuda2omp` is an AST-directed prototype compiler for a deliberately small CUDA
subset.  It asks Clang 17's CUDA frontend to parse and type-check the input,
then emits inspectable C++20 coroutine source.  No CUDA SDK or GPU is needed.

## Build and use

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./cuda2omp examples/reverse.cu -o reverse.cpp
clang++ -std=c++20 -fopenmp -Iruntime reverse.cpp -o reverse
./reverse
```

The emitted `.cpp` is the requested inspectable intermediate form. Set `CLANG`
or pass `--clang` to select a compatible Clang with CUDA parsing support.

## Execution model and implementation map

* `cuda2omp` consumes Clang's JSON AST. CUDA attributes identify kernels and
  device functions; resolved `DeclRefExpr`/`MemberExpr` nodes identify launches,
  calls, builtins, barriers, and shared declarations. It builds a direct call
  graph and computes the transitive barrier-capable fixed point.
* Barrier-capable code (including nested device calls) is emitted as composable
  C++20 `Task<T>` coroutines. `co_await` preserves caller/callee frames, locals,
  return continuations, and live values. Barrier-free functions currently use
  the same representation for simplicity; the analysis printed by the tool is
  ready to support the normal-function optimization.
* `runtime/cuda2omp_runtime.hpp` owns one coroutine per logical CUDA thread.
  `BlockScheduler` runs each to suspension, verifies every logical thread has
  arrived, then resumes the saved leaf coroutine handles. It never maps a CUDA
  barrier to an OpenMP barrier.
* A generated `Shared_<kernel>` object is constructed once per block and passed
  to every logical thread. CUDA indices live in an explicit `ThreadContext`.
* `launch` uses `#pragma omp parallel for` only over blocks. Scheduler and shared
  state are block-local, so concurrent blocks do not share bookkeeping.

## Supported subset

One-dimensional `<<<grid, block>>>` launches; `.x` of `threadIdx`, `blockIdx`,
`blockDim`, and `gridDim`; fixed-size kernel-local shared arrays; global/device
functions; nested device calls; barriers in kernels or device callees; ordinary
C++ expressions, conditionals, loops, and returns. Kernel arguments must remain
valid until the synchronous launch returns.

## Unsupported and known limitations

Dynamic shared memory, multidimensional launches, streams/asynchrony, atomics,
warp intrinsics, cooperative groups, textures, graphs, device allocation,
dynamic parallelism, function pointers/recursion, templates, overloaded CUDA
functions, and general CUDA runtime APIs are unsupported. Launch configuration
must contain exactly two expressions. Shared storage is presently supported
only when declared directly in a kernel. A legal barrier must be reached exactly
once per phase by every logical thread; otherwise the runtime emits a divergent
barrier diagnostic and aborts. This is a source prototype rather than a complete
Clang plugin: semantic selection comes from the CUDA AST, while source ranges
preserve user expressions in generated C++.
