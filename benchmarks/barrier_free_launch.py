#!/usr/bin/env python3
"""Compare the old all-coroutine launch shape with ordinary barrier-free calls."""
import pathlib, subprocess, tempfile, textwrap

ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=r'''
#include "cuda2omp_runtime.hpp"
#include <chrono>
#include <cstdio>
using namespace cuda2omp;
constexpr unsigned grid=256, block=256, repeats=100;
struct Shared {};
void ordinary(ThreadContext& ctx, Shared&, unsigned long long& sum) { sum += ctx.blockIdx_x*ctx.blockDim_x+ctx.threadIdx_x; }
Task<void> coroutine(ThreadContext& ctx, Shared&, unsigned long long& sum) { sum += ctx.blockIdx_x*ctx.blockDim_x+ctx.threadIdx_x; co_return; }
template<class F> double measure(F factory) {
  unsigned long long sum=0; auto begin=std::chrono::steady_clock::now();
  for(unsigned i=0;i<repeats;++i) launch<Shared>(grid,block,[&](auto& c,auto& s){return factory(c,s,sum);});
  auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  if(!sum) std::abort(); return elapsed;
}
int main() {
  double before=measure(coroutine), after=measure(ordinary);
  std::printf("barrier-free launch: before(all Task)=%.6fs after(selective)=%.6fs speedup=%.2fx\n",before,after,before/after);
}
'''
with tempfile.TemporaryDirectory() as temporary:
    directory=pathlib.Path(temporary); source=directory/'benchmark.cpp'; program=directory/'benchmark'
    source.write_text(textwrap.dedent(SOURCE))
    subprocess.run(['clang++','-O3','-std=c++20','-I',str(ROOT/'runtime'),str(source),'-o',str(program)],check=True)
    subprocess.run([str(program)],check=True)
