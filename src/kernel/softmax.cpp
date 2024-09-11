#include <stdio.h>

#include "kernel/my_math.h"

namespace jottings {

template <typename T>
__global__ void SoftmaxCudaKernel(const T *x, T *out, const int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = expf(x[idx]);
  }

  // TODO: use all-reduce to sum
  __shared__ T sum[1];
  if (0 == idx) {
    T temp = 0;
    for (int i = 0; i < n; ++i) {
      temp += out[i];
    }
    sum[0] = temp;
  }

  __syncthreads();

  if (idx < n) {
    out[idx] = out[idx] / sum[0];
  }
}

// only support one
template <typename T>
void SoftmaxCuda(const T *x, T *out, const int n) {
  const int blockSize = 256;  // threads per block
  const int numBlocks = (n + blockSize - 1) / blockSize;
  SoftmaxCudaKernel<<<numBlocks, blockSize>>>(x, out, n);
}

template void SoftmaxCuda<float>(const float *x, float *out, const int n);

}  // namespace jottings
