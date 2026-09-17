# cuda2omp prototype

`cuda2omp` is an AST-directed prototype compiler for a deliberately small CUDA
subset.  It asks Clang 17's CUDA frontend to parse and type-check the input,
then emits inspectable C++20 coroutine source.  No CUDA SDK or GPU is needed.

## Build and use

```sh
make
make test
./cuda2omp examples/reverse.cu -o reverse.cpp
clang++ -std=c++20 -fopenmp -Iruntime reverse.cpp -o reverse
./reverse
```

The existing compiler remains the `cuda2omp` Python executable while its
replacement is developed. When LLVM and Clang development packages are
available, `make` directly builds `build/cuda2omp-tool`, the C++ LibTooling
parsing and rewrite layer. The build uses `llvm-config` for compiler and linker
flags and does not require CMake. Use `make install` to install the available
tools and runtime header under `/usr/local`, or override `PREFIX`, `DESTDIR`,
`CXX`, or `LLVM_CONFIG` for another location or a staged package.

## Native parsing and rewrite layer

`cuda2omp-tool` obtains its compilation database with
`CommonOptionsParser`/`ClangTool` and accepts normal LibTooling arguments. For
example, the following parses one CUDA translation unit and exercises its
resolved-reference rewrite primitive:

```sh
build/cuda2omp-tool -o rewritten.cu --rename=old_name=new_name input.cu -- \
  -std=c++20 -x cuda --cuda-host-only -nocudainc -nocudalib \
  -include "$PWD/runtime/cuda_frontend_shim.hpp"
```

Every prospective edit is checked using its Clang spelling location. The tool
only rewrites tokens spelled in the main file and diagnoses CUDA declarations
from headers, macro expansions, invalid locations, and non-rewritable ranges.
This prevents byte offsets from an include or macro expansion from being
applied to the main file (and also makes UTF-8 source safe). The Python
transformer is intentionally still the production entry point until native
lowering reproduces all existing examples; `--rename` is an integration
primitive rather than the CUDA lowering interface.

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

One-dimensional `<<<grid, block>>>` and `cudaLaunchKernel` launches; `.x` of
`threadIdx`, `blockIdx`,
`blockDim`, and `gridDim`; fixed-size kernel-local shared arrays; global/device
functions; nested device calls; barriers in kernels or device callees; ordinary
C++ expressions, conditionals, loops, and returns. Kernel arguments must remain
valid until the synchronous launch returns.

## Unsupported and known limitations

Dynamic shared memory, multidimensional launches, streams/asynchrony, atomics,
warp intrinsics, cooperative groups, textures, graphs, device allocation,
dynamic parallelism, function pointers/recursion, templates, overloaded CUDA
functions, and general CUDA runtime APIs (apart from `cudaLaunchKernel`) are
unsupported. Launch configuration
must contain exactly two expressions. Shared storage is presently supported
only when declared directly in a kernel. A legal barrier must be reached exactly
once per phase by every logical thread; otherwise the runtime emits a divergent
barrier diagnostic and aborts. This is a source prototype rather than a complete
Clang plugin: semantic selection comes from the CUDA AST, while source ranges
preserve user expressions in generated C++.
