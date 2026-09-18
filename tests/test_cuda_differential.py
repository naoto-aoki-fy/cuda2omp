#!/usr/bin/env python3
"""Optional native-CUDA comparison; absence of a CUDA runner is a clean skip."""
import os, pathlib, shutil, subprocess, tempfile, textwrap, unittest
from support import CXX, ROOT

class CudaDifferential(unittest.TestCase):
 def test_vector_increment_matches_native_cuda(self):
  nvcc=shutil.which('nvcc')
  if not nvcc or os.environ.get('CUDA2OMP_CUDA_RUNNER') != '1':
   self.skipTest('CUDA runner unavailable (set CUDA2OMP_CUDA_RUNNER=1 with nvcc/GPU)')
  code=textwrap.dedent(r'''
   #include <cstdio>
   __global__ void add(int *p) { p[threadIdx.x] += int(threadIdx.x); }
   int main(){int h[4]={3,3,3,3},*d;cudaMalloc(&d,sizeof h);cudaMemcpy(d,h,sizeof h,cudaMemcpyHostToDevice);
    add<<<1,4>>>(d);cudaDeviceSynchronize();cudaMemcpy(h,d,sizeof h,cudaMemcpyDeviceToHost);
    for(int x:h)std::printf("%d ",x);cudaFree(d);}
  ''')
  with tempfile.TemporaryDirectory() as temporary:
   d=pathlib.Path(temporary); source=d/'input.cu'; source.write_text(code)
   native=d/'native'; subprocess.run([nvcc,str(source),'-o',str(native)],check=True)
   expected=subprocess.run([str(native)],check=True,text=True,capture_output=True).stdout
   # Host compatibility uses the same source while preserving explicit copies.
   out=d/'out.cpp'; translated=subprocess.run([str(ROOT/'cuda2omp'),str(source),'-o',str(out)],text=True,capture_output=True)
   self.assertEqual(translated.returncode,0,translated.stderr)
   host=d/'host'; compiled=subprocess.run([CXX,'-std=c++20','-fopenmp','-I',str(ROOT/'runtime'),str(out),'-o',str(host)],text=True,capture_output=True)
   self.assertEqual(compiled.returncode,0,compiled.stderr)
   actual=subprocess.run([str(host)],check=True,text=True,capture_output=True).stdout
   self.assertEqual(actual,expected)

if __name__ == '__main__': unittest.main(verbosity=2)
