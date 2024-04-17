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
