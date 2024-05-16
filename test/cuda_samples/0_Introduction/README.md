# Introduction

## async_api

该代码主要描述了cuda的异步特性，记录了gpu运行时间和cpu等待时间，其中，gpu运行时间通过`cudaEventElapsedTime`来统计。

注意，这里必须要在一个stream流上进行计算和统计。

## clock

通过reduce的方式来统计每个block的时间，最后计算每个block上花费的平均时间。

这里`clock_t`可以作为参数传入函数。

## concurrent_kernels

该段代码通过创建n个控制流，来实现n个kernel函数并发执行。

这里还通过reduce的方式进行求和。

## cpp_integration

这里应该主要是在cuda代码中使用了cpp自定义的数据结构int2，说明cuda代码中也可以很好地兼容cpp特性。

## cpp_overload

cuda中也支持函数重载，通过函数指针的方式，实现具体函数调用。

## cpp11_cuda

cuda中实现cpp11的特性。

## cuda_openmp

cuda中实现openmp。

## fp16_scalar_product

用fp16实现标量乘法计算。分别提供了naive和intrinsics的实现，并在最后做了精度比较，唯一比较可惜的是没有比较性能上差异，可以后期进行补充。

## matrix_mul

cuda实现矩阵乘。

## matrix_mul_driver

先将cuda函数编译生成fatbin文件，再在运行时加载该函数，下面是gpt上关于使用这种方式的一些优点：

1. 简化构建流程：使用fatbin文件可以简化构建流程，尤其是对于大型项目或者需要频繁修改源代码的情况。因为fatbin文件可以提前编译好，并且在链接阶段直接引用，而不需要重新编译。

2. 减少编译时间：将CUDA源代码编译成fatbin文件后，可以减少后续编译的时间，尤其是在重复构建时。这对于大型项目或者迭代开发非常有益。

3. 隐藏实现细节：使用fatbin文件可以隐藏实现细节，只需要将二进制文件链接到主程序中即可，而无需暴露源代码。这对于保护知识产权或者保护代码的安全性是有帮助的。

4. 便于跨平台部署：fatbin文件是与平台无关的二进制格式，因此可以轻松地在不同的CUDA支持的平台上部署和运行，而不必担心编译器差异或依赖关系。

总的来说，使用fatbin文件可以提高开发效率、简化构建流程、加快编译速度，同时也可以提高代码的安全性和可移植性。

## unified_memory_streams

一个简单的任务消费者程序，使用了CUDA的流（streams）和统一内存（Unified Memory）来执行任务。

## simple_AW_barrier

这段代码是一个 CUDA 示例程序，用于展示如何使用 CUDA 的 Arrive-Wait Barrier (AWBarrier) 来进行向量归一化操作。归一化过程通过计算两个向量的点积，并利用该点积的平方根来归一化这两个向量。

这段代码在我的环境无法编译，在cuda论坛上同样也看到这个[BUG](https://forums.developer.nvidia.com/t/cuda-12-1-error-when-building-cuda-samples/246465)了，看上去是编译器支持的问题，不知道后面是不是会修复，暂时移除测试。

## simple_assert

在cuda函数中实现assert。
