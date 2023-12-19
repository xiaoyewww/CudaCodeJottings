#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "profiler/profiler.h"

#define CUDA_CHECK(status)                                                     \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      throw std::runtime_error("cuda failure: " + std::to_string(ret) + " (" + \
                               cudaGetErrorString(ret) + ")" + "at " +         \
                               __FILE__ + ":" + std::to_string(__LINE__));     \
    }                                                                          \
  } while (0)

// no check for indices
__global__ void AddCuda(
    const float *x, const float *y, float *z, const int nx, const int ny) {
  int ix = threadIdx.x + blockDim.x * blockIdx.x;
  int iy = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = ix + iy * nx;
  // printf(
  //     "nx: %d, ny: %d, block dim(%d, %d)\n block id(%d, %d), thread id(%d, "
  //     "%d)\nixy(%d, %d), "
  //     "idx: %d \n",
  //     nx,
  //     ny,
  //     blockDim.x,
  //     blockDim.y,
  //     blockIdx.x,
  //     blockIdx.y,
  //     threadIdx.x,
  //     threadIdx.y,
  //     ix,
  //     iy,
  //     idx);
  if (ix < nx && iy < ny) {
    z[idx] = x[idx] + y[idx];
    // printf("idx %d is ready: %f, %f, %f\n", idx, z[idx], x[idx], y[idx]);
  }
}

void AddCpu(const float *x, const float *y, float *z, const int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

int main() {
  int dev = 0;
  cudaDeviceProp devProp;
  CUDA_CHECK(cudaGetDeviceProperties(&devProp, dev));
  std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
  std::cout << "SM的数量: " << devProp.multiProcessorCount << std::endl;
  std::cout << "每个线程块的共享内存大小："
            << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
  std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock
            << std::endl;
  std::cout << "每个EM的最大线程数: " << devProp.maxThreadsPerMultiProcessor
            << std::endl;
  std::cout << "每个SM的最大线程束数: "
            << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

  constexpr int kCols = 1 << 16;
  constexpr int kRows = 1 << 12;
  constexpr int kNums = kCols * kRows;
  constexpr int nBytes = kCols * kRows * sizeof(float);
  // malloc host memory
  float *x, *y, *z, *z_res;
  x = (float *)malloc(nBytes);
  y = (float *)malloc(nBytes);
  z = (float *)malloc(nBytes);
  z_res = (float *)malloc(nBytes);

  PROFILER_FUNC(AddCpu, x, y, z_res, kNums);

  // malloc device memory
  float *d_x, *d_y, *d_z;
  CUDA_CHECK(cudaMalloc((void **)&d_x, nBytes));
  CUDA_CHECK(cudaMalloc((void **)&d_y, nBytes));
  CUDA_CHECK(cudaMalloc((void **)&d_z, nBytes));
  CUDA_CHECK(
      cudaMemcpy((void *)d_x, (void *)x, nBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy((void *)d_y, (void *)y, nBytes, cudaMemcpyHostToDevice));

  // block size and grid size
  const unsigned int block_x = 1024;
  const unsigned int block_y = 1024;
  dim3 block(block_x, block_y);
  std::cout << "block: " << block.x << "," << block.y << std::endl;

  const unsigned int grid_x = (kCols + block.x - 1) / block.x;
  const unsigned int grid_y = (kRows + block.y - 1) / block.y;
  dim3 grid(grid_x, grid_y);
  std::cout << "grid: " << grid.x << "," << grid.y << std::endl;
  // std::cout << "kCols:" << kCols << std::endl;
  // std::cout << "kRows:" << kRows << std::endl;
  // std::cout << "grid:" << grid.x << " " << grid.y << std::endl;
  // compute
  // AddCuda<<<grid, block>>>(d_x, d_y, d_z, kCols, kRows);

  PROFILER_CUDA_FUNC(AddCuda, grid, block, d_x, d_y, d_z, kCols, kRows);
  cudaDeviceSynchronize();
  // if no this line ,it can not output hello world from gpu
  // but it will reset the device memory
  // cudaDeviceReset();

  cudaMemcpy((void *)z, (void *)d_z, nBytes, cudaMemcpyDeviceToHost);

  // check
  float max_error = 0.0;
  for (int i = 0; i < kNums; ++i) {
    // if (fabs(z[i] - z_res[i]) > 1) {
    //   std::cout << "z[i]: " << z[i] << std::endl;
    //   std::cout << "z_res[i]: " << z_res[i] << std::endl;
    //   std::cout << i << std::endl;
    // }
    max_error = fmax(max_error, fabs(z[i] - z_res[i]));
  }
  std::cout << "max_error is: " << max_error << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  free(x);
  free(y);
  free(z);
  free(z_res);
  return 0;
}