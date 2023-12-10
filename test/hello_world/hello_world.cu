#include <stdio.h>

#include "profiler/profiler.h"

__global__ void hello_world(void) { printf("GPU: Hello world!\n"); }

void HelloWord() { printf("CPU: Hello world!\n"); }

int main(int argc, char **argv) {
  PROFILER_FUNC(HelloWord);
  PROFILER_FUNC_1000TIMES(HelloWord);
  PROFILER_CUDA_FUNC(hello_world, 1, 10);
  PROFILER_CUDA_FUNC_1000TIMES(hello_world, 1, 10);
  cudaDeviceReset();  // if no this line ,it can not output hello world from gpu
  return 0;
}
