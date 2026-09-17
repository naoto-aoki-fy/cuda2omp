#!/usr/bin/env python3
import pathlib, subprocess, tempfile, textwrap, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]

class Tests(unittest.TestCase):
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

if __name__=='__main__': unittest.main()
