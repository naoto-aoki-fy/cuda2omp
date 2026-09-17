#include <iostream>

__global__ void reverse_block(int *x) {
  __shared__ int tmp[64];
  int tid = threadIdx.x;
  tmp[tid] = x[blockIdx.x * blockDim.x + tid];
  __syncthreads();
  x[blockIdx.x * blockDim.x + tid] = tmp[blockDim.x - 1 - tid];
}

int main() {
  int x[128]; for (int i=0;i<128;++i) x[i]=i;
  reverse_block<<<2,64>>>(x);
  for (int b=0;b<2;++b) for(int i=0;i<64;++i)
    if (x[b*64+i] != b*64+63-i) return 1;
  std::cout << "ok\n";
}
