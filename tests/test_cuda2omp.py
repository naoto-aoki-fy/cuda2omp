#!/usr/bin/env python3
import json, os, pathlib, shlex, subprocess, tempfile, textwrap, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]

class Tests(unittest.TestCase):
 def test_staged_install_is_self_contained(self):
  with tempfile.TemporaryDirectory() as temporary:
   stage=pathlib.Path(temporary)/'stage'; work=pathlib.Path(temporary)/'work'
   work.mkdir()
   install=subprocess.run(['make','install','PREFIX=/usr','DESTDIR='+str(stage)],
                          cwd=ROOT,text=True,capture_output=True)
   self.assertEqual(install.returncode,0,install.stderr)
   driver=stage/'usr/bin/cuda2omp'
   resource=stage/'usr/lib/cuda2omp/cuda_frontend_shim.hpp'
   public=stage/'usr/include/cuda2omp_runtime.hpp'
   compatibility=stage/'usr/include/cuda_runtime.h'
   self.assertTrue(driver.is_file()); self.assertTrue(resource.is_file()); self.assertTrue(public.is_file())
   self.assertTrue(compatibility.is_file())
   source=work/'input.cu'; output=work/'output.cpp'; executable=work/'program'
   source.write_text('__global__ void set(int*x){x[threadIdx.x]=7;} '
                     'int main(){int x[1]={};set<<<1,1>>>(x);return x[0]!=7;}\n')
   translated=subprocess.run([str(driver),'-v',str(source),'-o',str(output)],cwd=work,
                             text=True,capture_output=True)
   self.assertEqual(translated.returncode,0,translated.stderr)
   self.assertNotIn(str(ROOT),translated.stderr)
   self.assertNotIn(str(ROOT),output.read_text())
   compiled=subprocess.run(['clang++','-std=c++20','-I',str(stage/'usr/include'),
                            str(output),'-o',str(executable)],cwd=work,
                           text=True,capture_output=True)
   self.assertEqual(compiled.returncode,0,compiled.stderr)
   run=subprocess.run([str(executable)],cwd=work,text=True,capture_output=True)
   self.assertEqual(run.returncode,0,run.stderr)
   env=os.environ.copy(); env['CUDA2OMP_RESOURCE_DIR']=str(work/'missing')
   missing=subprocess.run([str(driver),str(source),'-o',str(output)],cwd=work,
                          env=env,text=True,capture_output=True)
   self.assertNotEqual(missing.returncode,0)
   self.assertIn("required frontend resource 'cuda_frontend_shim.hpp' not found",missing.stderr)
   self.assertIn(str(work/'missing'),missing.stderr)
   # An explicit option has priority over the environment override.
   explicit=subprocess.run([str(driver),'--resource-dir',str(stage/'usr/lib/cuda2omp'),
                            str(source),'-o',str(output)],cwd=work,env=env,
                           text=True,capture_output=True)
   self.assertEqual(explicit.returncode,0,explicit.stderr)

 def test_cpu_cuda_runtime_api_and_error_state(self):
  code=r'''
   #include <cuda_runtime.h>
   #include <cstring>
   #include <thread>
   int main() {
     static_assert(cudaSuccess == 0);
     dim3 extent(2, 3); if (extent.x != 2 || extent.y != 3 || extent.z != 1) return 1;
     int source[3]={4,5,6}; int *copy=nullptr;
     if (cudaMalloc(reinterpret_cast<void **>(&copy), sizeof source) != cudaSuccess) return 2;
     if (cudaMemcpy(copy, source, sizeof source, cudaMemcpyHostToDevice) != cudaSuccess) return 3;
     if (std::memcmp(copy, source, sizeof source)) return 4;
     if (cudaMemset(copy, 0, sizeof source) != cudaSuccess || copy[1] != 0) return 5;
     if (cudaDeviceSynchronize() != cudaSuccess || cudaFree(copy) != cudaSuccess) return 6;
     if (cudaMemcpy(nullptr, source, 1, cudaMemcpyHostToHost) != cudaErrorInvalidValue) return 7;
     if (cudaPeekAtLastError() != cudaErrorInvalidValue) return 8;
     cudaError_t thread_error=cudaSuccess;
     std::thread worker([&]{ cudaMemset(nullptr, 0, 1); thread_error=cudaPeekAtLastError(); });
     worker.join();
     if (thread_error != cudaErrorInvalidValue || cudaPeekAtLastError() != cudaErrorInvalidValue) return 9;
     if (cudaGetLastError() != cudaErrorInvalidValue || cudaPeekAtLastError() != cudaSuccess) return 10;
     if (cudaMemcpy(nullptr, nullptr, 0, (cudaMemcpyKind)99) != cudaErrorInvalidMemcpyDirection) return 11;
     if (std::strcmp(cudaGetErrorString(cudaErrorMemoryAllocation), "out of memory")) return 12;
   }
  '''
  with tempfile.TemporaryDirectory() as temporary:
   directory=pathlib.Path(temporary); source=directory/'api.cpp'; program=directory/'api'
   source.write_text(textwrap.dedent(code))
   compiled=subprocess.run(['clang++','-std=c++20','-pthread','-I',str(ROOT/'runtime'),
                            str(source),'-o',str(program)],text=True,capture_output=True)
   self.assertEqual(compiled.returncode,0,compiled.stderr)
   run=subprocess.run([str(program)],text=True,capture_output=True)
   self.assertEqual(run.returncode,0,run.stderr)

 def test_cpu_cuda_runtime_rejects_invalid_and_overflowing_operations(self):
  code=r'''
   #include <cuda_runtime.h>
   #include <stdint.h>
   int main() {
     char value=1; void *allocation=&value;
     if (cudaMalloc(nullptr, 1) != cudaErrorInvalidValue) return 1;
     if (cudaMalloc(&allocation, (size_t)PTRDIFF_MAX + 1) != cudaErrorInvalidValue) return 2;
     if (allocation != &value) return 3;
     if (cudaMemset(nullptr, 0, 1) != cudaErrorInvalidValue) return 4;
     if (cudaMemcpy(&value, nullptr, 1, cudaMemcpyDefault) != cudaErrorInvalidValue) return 5;
     if (cudaMemset(nullptr, 0, 0) != cudaSuccess) return 6;
     if (cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyHostToHost) != cudaSuccess) return 7;
   }
  '''
  with tempfile.TemporaryDirectory() as temporary:
   directory=pathlib.Path(temporary); source=directory/'negative.cpp'; program=directory/'negative'
   source.write_text(textwrap.dedent(code))
   compiled=subprocess.run(['clang++','-std=c++20','-I',str(ROOT/'runtime'),str(source),
                            '-o',str(program)],text=True,capture_output=True)
   self.assertEqual(compiled.returncode,0,compiled.stderr)
   run=subprocess.run([str(program)],text=True,capture_output=True)
   self.assertEqual(run.returncode,0,run.stderr)

 def test_frontend_shim_uses_public_runtime_declarations(self):
  shim=(ROOT/'runtime/cuda_frontend_shim.hpp').read_text()
  self.assertIn('#include "cuda_runtime.h"',shim)
  self.assertNotIn('struct dim3',shim)

 def test_compilation_database_arguments_and_explicit_override(self):
  with tempfile.TemporaryDirectory() as temporary:
   root=pathlib.Path(temporary)/'project with spaces'
   source_dir=root/'source'; build=root/'build'; includes=root/'relative includes'
   generated=build/'generated headers'
   for directory in (source_dir,build,includes,generated): directory.mkdir(parents=True,exist_ok=True)
   (includes/'config.hpp').write_text('#define FROM_RELATIVE_INCLUDE 1\n')
   (generated/'generated.hpp').write_text('#define FROM_GENERATED_INCLUDE 1\n')
   source=source_dir/'kernel file.cu'
   source.write_text(textwrap.dedent('''
    #include "config.hpp"
    #include "generated.hpp"
    #if VALUE != 9 || !FROM_RELATIVE_INCLUDE || !FROM_GENERATED_INCLUDE
    #error compiler arguments were not preserved or overridden
    #endif
    __global__ void kernel(int *x) { x[threadIdx.x] = VALUE; }
    int main() { int x[1]={}; kernel<<<1,1>>>(x); return x[0]==9?0:1; }
   '''))
   response=build/'flags.rsp'
   response.write_text(shlex.join(['-I../relative includes','-Igenerated headers','-DVALUE=7',
                                   '-std=c++17','-MF','discard.d','-c']))
   entry={'directory':str(build),'file':'../source/kernel file.cu',
          'arguments':['clang++','@flags.rsp','../source/kernel file.cu','-o','old output.o']}
   (build/'compile_commands.json').write_text(json.dumps([entry]))
   output=root/'result file.cpp'
   command=[str(ROOT/'cuda2omp'),'-p',str(build),'-v',str(source),'-o',str(output),
            '--','-DVALUE=9','-std=c++20']
   result=subprocess.run(command,text=True,capture_output=True)
   self.assertEqual(result.returncode,0,result.stderr)
   self.assertTrue(output.is_file())
   self.assertIn('frontend cwd='+str(build),result.stderr)
   invocation=next(line for line in result.stderr.splitlines()
                   if line.startswith('cuda2omp: frontend ') and ' cwd=' not in line)
   self.assertIn("'-I../relative includes'",invocation)
   self.assertIn("'-Igenerated headers'",invocation)
   self.assertIn('-DVALUE=9',invocation)
   self.assertNotIn('-DVALUE=7',invocation)
   self.assertNotIn('discard.d',invocation)
   self.assertNotIn('old output.o',invocation)

 def test_compilation_database_missing_and_ambiguous_diagnostics(self):
  with tempfile.TemporaryDirectory() as temporary:
   root=pathlib.Path(temporary); source=root/'input.cu'; source.write_text('int main(){}\n')
   database=root/'compile_commands.json'
   for entries,message in (([], 'no compile_commands.json entry'),
                           ([{'directory':str(root),'file':'input.cu','arguments':['clang++','input.cu']}]*2,
                            'ambiguous compile_commands.json entries')):
    database.write_text(json.dumps(entries))
    result=subprocess.run([str(ROOT/'cuda2omp'),'-p',str(root),str(source),'-o',str(root/'out.cpp')],
                          text=True,capture_output=True)
    self.assertNotEqual(result.returncode,0)
    self.assertIn(message,result.stderr)

 def test_native_frontend_contract(self):
  source=(ROOT/'native/cuda2omp_tool.cpp').read_text()
  makefile=(ROOT/'Makefile').read_text()
  self.assertIn('CommonOptionsParser::create',source)
  self.assertIn('ClangTool Tool',source)
  self.assertIn('SM.getSpellingLoc',source)
  self.assertIn('SM.isWrittenInMainFile',source)
  self.assertIn('Location.isMacroID()',source)
  self.assertIn('Rewrite.ReplaceText',source)
  self.assertIn('$(LLVM_CONFIG) --cxxflags',makefile)
  self.assertNotIn('cmake',makefile.lower())

 def test_native_utf8_and_location_fixture(self):
  fixture=ROOT/'tests/fixtures/native_locations.cu'
  text=fixture.read_text()
  self.assertIn('café 🚀',text)
  self.assertIn('namespace image',text)
  self.assertIn('READ_PIXEL(amount)',text)
  self.assertIn('Pixel amount',text)
  self.assertTrue((fixture.parent/'included_device.cuh').is_file())

  tool=ROOT/'build/cuda2omp-tool'
  if not tool.is_file():
   return
  with tempfile.TemporaryDirectory() as d:
   output=pathlib.Path(d)/'rewritten.cu'
   command=[str(tool),'-o',str(output),'--rename=amount=replacement',
            str(fixture),'--','-std=c++20','-x','cuda','--cuda-host-only',
            '-nocudainc','-nocudalib','-include',
            str(ROOT/'runtime/cuda_frontend_shim.hpp'),'-I',str(fixture.parent)]
   p=subprocess.run(command,text=True,capture_output=True)
   self.assertEqual(p.returncode,0,p.stderr)
   rewritten=output.read_text()
   self.assertIn('café 🚀',rewritten)
   self.assertIn('from_header(replacement)',rewritten)
   # A reference spelled through a macro is diagnosed and its definition is
   # never changed as an accidental numeric-offset edit.
   self.assertIn('#define READ_PIXEL(pixel) ((pixel).value)',rewritten)
   self.assertIn('macro expansion',p.stderr)
   self.assertIn('included file; not editing',p.stderr)

 def run_cuda(self, code):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d); cu=d/'in.cu'; cpp=d/'out.cpp'; exe=d/'a.out'; cu.write_text(textwrap.dedent(code))
   p=subprocess.run([str(ROOT/'cuda2omp'),str(cu),'-o',str(cpp)],text=True,capture_output=True)
   self.assertEqual(p.returncode,0,p.stderr)
   p=subprocess.run(['clang++','-std=c++20','-fopenmp','-I',str(ROOT/'runtime'),str(cpp),'-o',str(exe)],text=True,capture_output=True)
   if p.returncode: # Swift clang commonly has no OpenMP runtime; pragmas remain valid without it.
    p=subprocess.run(['clang++','-std=c++20','-I',str(ROOT/'runtime'),str(cpp),'-o',str(exe)],text=True,capture_output=True)
   self.assertEqual(p.returncode,0,p.stderr+'\n'+cpp.read_text())
   p=subprocess.run([str(exe)],text=True,capture_output=True)
   self.assertEqual(p.returncode,0,p.stdout+p.stderr)

 def assert_rejected(self, code, expected):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d); cu=d/'in.cu'; cpp=d/'out.cpp'
   cu.write_text(textwrap.dedent(code)); cpp.write_text('sentinel')
   p=subprocess.run([str(ROOT/'cuda2omp'),str(cu),'-o',str(cpp)],text=True,capture_output=True)
   self.assertNotEqual(p.returncode,0,p.stderr)
   self.assertIn(expected,p.stderr)
   self.assertRegex(p.stderr,rf'{cu}:\d+:\d+: (?:error: )?')
   self.assertEqual(cpp.read_text(),'sentinel','validation modified the output file')

 def test_validation_rejections(self):
  cases={
   'builtin member': ('__global__ void k(){int i=threadIdx.y;} int main(){k<<<1,1>>>();}', 'only .x'),
   'template': ('template<class T> __device__ T f(T x){return x;} __global__ void k(){}', 'templates are unsupported'),
   'indirect call': ('__device__ int f(int x){return x;} __global__ void k(){int(*p)(int)=f; p(1);}', 'indirect calls'),
   'recursion': ('__device__ int f(int x){return x?f(x-1):0;} __global__ void k(){f(1);}', 'recursive CUDA call'),
   'macro range': ('#define TID threadIdx.x\n__global__ void k(){int x=TID;}', 'macro range'),
   'dynamic shared': ('__global__ void k(){extern __shared__ int x[];}', 'fixed-size kernel-local'),
   'stream': ('__global__ void k(){} int main(){k<<<1,1,0,(void*)1>>>();}', 'only <<<grid, block>>>'),
   'device variable': ('__device__ int value; __global__ void k(){}', 'device/global CUDA variables'),
   'shared in device': ('__device__ void f(){__shared__ int x[2];} __global__ void k(){f();}', 'directly kernel-local'),
   'shared scalar': ('__global__ void k(){__shared__ int x;}', 'fixed-size kernel-local'),
   'unsupported expression': ('__global__ void k(int x){switch(x){case 1: break;}}', 'unsupported CUDA expression SwitchStmt'),
   'unsupported target': ('__device__ int other(int); __global__ void k(){other(1);}', "call target 'other' is unsupported"),
   'launch dimensions': ('__global__ void k(){} int main(){k<<<1,1,0>>>();}', 'only <<<grid, block>>>'),
  }
  for label,(code,message) in cases.items():
   with self.subTest(label=label): self.assert_rejected(code,message)

 def test_cuda_declaration_in_header_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d); header=d/'device.cuh'; cu=d/'in.cu'; out=d/'out.cpp'
   header.write_text('__device__ int from_header(){return 1;}\n')
   cu.write_text('#include "device.cuh"\n__global__ void k(){from_header();}\n')
   p=subprocess.run([str(ROOT/'cuda2omp'),str(cu),'-o',str(out)],text=True,capture_output=True)
   self.assertNotEqual(p.returncode,0,p.stderr)
   self.assertIn('non-main file',p.stderr)
   self.assertFalse(out.exists())

 def test_all_semantics(self):
  self.run_cuda(r'''
   __device__ int square(int x) { return x*x; }
   __device__ int across_barrier(int *tmp, int tid, int v) {
     int saved=v+3; tmp[tid]=saved; __syncthreads(); return tmp[7-tid]+saved;
   }
   __global__ void basic(int*x) { int i=blockIdx.x*blockDim.x+threadIdx.x; x[i]=square(x[i]); }
   __global__ void reverse(int*x) { __shared__ int tmp[8]; int t=threadIdx.x;
     tmp[t]=x[blockIdx.x*blockDim.x+t]; if(t&1) { int q=1; tmp[t]+=q; } else tmp[t]+=1;
     __syncthreads(); x[blockIdx.x*blockDim.x+t]=tmp[7-t]; }
   __global__ void nested(int*x) { __shared__ int tmp[8]; int t=threadIdx.x; int before=x[t];
     x[t]=across_barrier(tmp,t,before)+before; }
   int main(){int a[16];for(int i=0;i<16;++i)a[i]=i+1;basic<<<2,8>>>(a);
     for(int i=0;i<16;++i)if(a[i]!=(i+1)*(i+1))return 1;
     int b[16];for(int i=0;i<16;++i)b[i]=i;reverse<<<2,8>>>(b);
     for(int k=0;k<2;++k)for(int i=0;i<8;++i)if(b[k*8+i]!=k*8+8-i)return 2;
     int c[8];for(int i=0;i<8;++i)c[i]=i;nested<<<1,8>>>(c);
     for(int i=0;i<8;++i)if(c[i] != ((7-i)+3)+(i+3)+i)return 3; }
  ''')

 def test_cuda_launch_kernel(self):
  self.run_cuda(r'''
   __global__ void fill(int *x, int value) {
     x[blockIdx.x*blockDim.x+threadIdx.x] = value;
   }
   int main() {
     int x[8] = {}; int *output=x; int value=42;
     void *args[] = {&output, &value};
     int status=cudaLaunchKernel((const void*)fill, 2, 4, args, 0, nullptr);
     if(status != 0) return 1;
     for(int v:x) if(v != 42) return 2;
   }
  ''')

 def test_canonical_declarations_disambiguate_names_and_overloads(self):
  self.run_cuda(r'''
   int adjust(double x) { return int(x)+100; }
   namespace left {
   __device__ int adjust(int x) { return x+1; }
   __global__ void same(int *x) { __shared__ int tmp[2];
     int t=threadIdx.x; tmp[t]=adjust(x[t]); __syncthreads();
     { int tmp=40; if(t==0) x[2]=tmp; }
     x[t]=tmp[1-t]; }
   }
   namespace right {
   __device__ long adjust(long x) { return x+10; }
   __global__ void same(long *x) { __shared__ long tmp[2];
     int t=threadIdx.x; tmp[t]=adjust(x[t]); __syncthreads(); x[t]=tmp[1-t]; }
   }
   int main() {
     int a[3]={1,2,0}; long b[2]={3,4};
     left::same<<<1,2>>>(a); right::same<<<1,2>>>(b);
     if(a[0]!=3 || a[1]!=2 || a[2]!=40) return 1;
     if(b[0]!=14 || b[1]!=13) return 2;
     if(adjust(1.0)!=101) return 3;
   }
  ''')

 def test_generated_identifiers_are_stable_and_collision_free(self):
  code=r'''
   int cuda2omp_fn_0 = 7;
   namespace a { __global__ void kernel(int *x) { x[0]=1; } }
   namespace b { __global__ void kernel(int *x) { x[0]=2; } }
   int main(){ int x=0; a::kernel<<<1,1>>>(&x); if(x!=1)return 1;
     b::kernel<<<1,1>>>(&x); return x==2?0:2; }
  '''
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d); source=d/'in.cu'; first=d/'one.cpp'; second=d/'two.cpp'
   source.write_text(textwrap.dedent(code))
   for output in (first,second):
    p=subprocess.run([str(ROOT/'cuda2omp'),str(source),'-o',str(output)],text=True,capture_output=True)
    self.assertEqual(p.returncode,0,p.stderr)
   self.assertEqual(first.read_text(),second.read_text())
   transformed=first.read_text()
   self.assertIn('cuda2omp_fn_1',transformed)
   self.assertIn('cuda2omp_fn_2',transformed)
   self.assertIn('cuda2omp_shared_0',transformed)
   self.assertIn('cuda2omp_shared_1',transformed)

 def test_in_place_declarations_and_lexical_contexts_compile(self):
  self.run_cuda(r'''
   typedef int Number;
   __device__ Number later(Number);
   __device__ Number earlier(Number x) { return later(x)+1; }
   __device__ Number later(Number x) { return x*2; }

   namespace named {
   struct Payload { Number value; };
   __global__ void apply(Payload *p) { p->value=earlier(p->value); }
   }

   namespace {
   __device__ Number hidden(Number x);
   __device__ Number hidden(Number x) { return x+3; }
   struct Local { Number value; };
   __global__ void local(Local *p) { p->value=hidden(p->value); }
   }

   extern "C" {
   __device__ Number c_helper(Number);
   __device__ Number c_helper(Number x) { return x+4; }
   struct CValue { Number value; };
   __global__ void c_kernel(CValue *p) { p->value=c_helper(p->value); }
   }

   int main() {
     named::Payload a{5}; named::apply<<<1,1>>>(&a); if(a.value!=11)return 1;
     Local b{7}; local<<<1,1>>>(&b); if(b.value!=10)return 2;
     CValue c{8}; c_kernel<<<1,1>>>(&c); return c.value==12?0:3;
   }
  ''')

 def test_runtime_include_and_shared_state_stay_in_place(self):
  code='''// heading\n#pragma once\n#include <cstddef>\n\nnamespace scope {\nstruct JustBefore { int value; };\n__global__ void kernel(JustBefore *p) { __shared__ int tmp[1]; tmp[0]=p->value; p->value=tmp[0]+1; }\n}\nint main(){scope::JustBefore x{1};scope::kernel<<<1,1>>>(&x);return x.value==2?0:1;}\n'''
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d); cu=d/'in.cu'; cpp=d/'out.cpp'; cu.write_text(code)
   p=subprocess.run([str(ROOT/'cuda2omp'),str(cu),'-o',str(cpp)],text=True,capture_output=True)
   self.assertEqual(p.returncode,0,p.stderr)
   transformed=cpp.read_text()
   self.assertGreater(transformed.index('#include "cuda2omp_runtime.hpp"'),transformed.index('#include <cstddef>'))
   self.assertGreater(transformed.index('struct cuda2omp_shared_'),transformed.index('struct JustBefore'))
   self.assertLess(transformed.index('struct cuda2omp_shared_'),transformed.index('cuda2omp::Task<void>'))
   p=subprocess.run(['clang++','-std=c++20','-I',str(ROOT/'runtime'),str(cpp),'-o',str(d/'a.out')],text=True,capture_output=True)
   self.assertEqual(p.returncode,0,p.stderr+'\n'+transformed)

if __name__=='__main__': unittest.main()
