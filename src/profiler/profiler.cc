#include "profiler/profiler.h"

double CpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, nullptr);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
