#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

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

template <typename T>
void SoftmaxCpu(const T *x, T *output, const int n) {
  T sum = 0.f;
  for (int i = 0; i < n; ++i) {
    output[i] = std::exp(x[i]);
    sum += output[i];
  }
  for (int i = 0; i < n; ++i) {
    output[i] = output[i] / sum;
  }
}

// __shared__ clock_t *global_now;

__global__ void SoftmaxCuda(float *x, float *out, const int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("CPU: Hello world!\n");
  if (idx < n) {
    out[idx] = expf(x[idx]);
    x[idx] += 1.0f;
  }
  printf("ssa\n");
  // clock_t start = clock();
  // clock_t now;
  // for (;;) {
  //   now = clock();
  //   clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
  //   if (cycles >= 10000) {
  //     break;
  //   }
  // }
  // // Stored "now" in global memory here to prevent the
  // // compiler from optimizing away the entire loop.
  // *global_now = now;
}

template <typename T>
void PrintResults(T *nums, int n) {
  for (int i = 0; i < n; ++i) {
    std::cout << nums[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T, typename U>
bool CompareResults(const T *nums_cpu,
                    const T *nums_cuda,
                    const int n,
                    const float err = 1e-5) {
  for (int i = 0; i < n; ++i) {
    if (std::abs(nums_cpu[i] - nums_cuda[i]) > err) {
      return false;
    }
  }
  return true;
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Compute capability: " << prop.major << "." << prop.minor
            << std::endl;

  constexpr int kLen = 10;
  constexpr int kLenBytes = kLen * sizeof(float);
  std::cout << "kLen: " << kLen << std::endl;
  std::cout << "kLenBytes: " << kLenBytes << std::endl;
  // malloc host memory
  float *x, *output, *output_cuda, *x_cuda;
  x = (float *)malloc(kLenBytes);
  x_cuda = (float *)malloc(kLenBytes);
  output = (float *)malloc(kLenBytes);
  output_cuda = (float *)malloc(kLenBytes);

  for (int i = 0; i < kLen; ++i) {
    x[i] = 2.0 + i;
  }

  PROFILER_FUNC(SoftmaxCpu, x, output, kLen);

  PrintResults(output, kLen);

  // malloc device memory
  float *d_x, *d_output;
  CUDA_CHECK(cudaMalloc((void **)&d_x, kLenBytes));
  CUDA_CHECK(cudaMalloc((void **)&d_output, kLenBytes));
  CUDA_CHECK(
      cudaMemcpy((void *)d_x, (void *)x, kLenBytes, cudaMemcpyHostToDevice));

  jottings::SoftmaxCuda(d_x, d_output, kLen);
  PROFILER_FUNC(jottings::SoftmaxCuda, d_x, d_output, kLen);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
  }
  // cudaDeviceReset();
  // compare
  CUDA_CHECK(
      cudaMemcpy(output_cuda, d_output, kLenBytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemcpy(
      (void *)x_cuda, (void *)d_x, kLenBytes, cudaMemcpyDeviceToHost));

  PrintResults(output_cuda, kLen);

  assert((void("The cuda results are not right!"),
          CompareResults<float, float>(output_cuda, output, kLen)));

  free(x);
  free(output);
  free(output_cuda);

  cudaFree(d_x);
  cudaFree(d_output);
  return 0;
}