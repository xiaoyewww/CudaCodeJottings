#ifndef MATH_H
#define MATH_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

namespace jottings {

__global__ void AddCuda(
    const float *x, const float *y, float *z, const int nx, const int ny);

// TODO: support ND and parameter for axis
template <typename T>
void SoftmaxCuda(const T *x, T *out, const int n);

// template<typename T>
// __global__ void SoftmaxCuda(T *x, T *out, const int n);
}  // namespace jottings

#endif
