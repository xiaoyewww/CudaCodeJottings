#include "kernel/my_math.h"

namespace jottings {

template <typename T>
__global__ void AddCudaKernel(
    const T *x, const T *y, T *z, const int nx, const int ny) {
  int ix = threadIdx.x + blockDim.x * blockIdx.x;
  int iy = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = ix + iy * nx;

  if (ix < nx && iy < ny) {
    z[idx] = x[idx] + y[idx];
  }
}

// only support one
template <typename T>
void AddCuda(const T *x, const T *y, T *z, const int nx, const int ny) {
  // block size and grid size
  // TODO: how to opt the size of blocks and threads
  const unsigned int block_x = 256;
  const unsigned int block_y = 4;
  dim3 block(block_x, block_y);

  const unsigned int grid_x = (nx + block.x - 1) / block.x;
  const unsigned int grid_y = (ny + block.y - 1) / block.y;
  dim3 grid(grid_x, grid_y);

  AddCudaKernel<<<grid, block>>>(x, y, z, nx, ny);
}

template void AddCuda<float>(
    const float *x, const float *y, float *z, const int nx, const int ny);

}  // namespace jottings
