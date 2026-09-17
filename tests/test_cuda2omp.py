#!/usr/bin/env python3
import pathlib, subprocess, tempfile, textwrap, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]

class Tests(unittest.TestCase):
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
   'overload': ('__device__ int f(int x){return x;} __device__ int f(long x){return x;} __global__ void k(){}', 'overloaded CUDA function'),
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

if __name__=='__main__': unittest.main()
