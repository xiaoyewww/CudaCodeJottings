#ifndef MATH_H
#define MATH_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace jottings {

__global__ void AddCuda(
    const float *x, const float *y, float *z, const int nx, const int ny);
}

#endif
