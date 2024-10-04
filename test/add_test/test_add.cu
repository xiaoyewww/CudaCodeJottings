#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "all_in_one.h"

#define CUDA_CHECK(status)                                                     \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      throw std::runtime_error("cuda failure: " + std::to_string(ret) + " (" + \
                               cudaGetErrorString(ret) + ")" + "at " +         \
                               __FILE__ + ":" + std::to_string(__LINE__));     \
    }                                                                          \
  } while (0)

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

  constexpr int kCols = 1 << 12;
  constexpr int kRows = 1 << 8;
  constexpr int kNums = kCols * kRows;
  constexpr int nBytes = kCols * kRows * sizeof(float);
  // malloc host memory
  float *x, *y, *z, *z_res;
  x = (float *)malloc(nBytes);
  y = (float *)malloc(nBytes);
  z = (float *)malloc(nBytes);
  z_res = (float *)malloc(nBytes);

  for (int i = 0; i < kNums; ++i) {
    x[i] = static_cast<float>(i);
    y[i] = static_cast<float>(i);
  }

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

  PROFILER_FUNC(
      jottings::AddCuda, d_x, d_y, d_z, kCols, kRows);
  cudaDeviceSynchronize();
  // if no this line ,it can not output hello world from gpu
  // but it will reset the device memory
  // cudaDeviceReset();

  cudaMemcpy((void *)z, (void *)d_z, nBytes, cudaMemcpyDeviceToHost);

  // check
  float max_error = 0.0;
  for (int i = 0; i < 10; ++i) {
    if (fabs(z[i] - z_res[i]) > 1) {
      std::cout << "z[i]: " << z[i] << std::endl;
      std::cout << "z_res[i]: " << z_res[i] << std::endl;
      std::cout << i << std::endl;
    }
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