#include "kernel/math.h"

namespace jottings {

// no check for indices
__global__ void AddCuda(
    const float *x, const float *y, float *z, const int nx, const int ny) {
  int ix = threadIdx.x + blockDim.x * blockIdx.x;
  int iy = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = ix + iy * nx;

  if (ix < nx && iy < ny) {
    z[idx] = x[idx] + y[idx];
  }
}
}  // namespace jottings
