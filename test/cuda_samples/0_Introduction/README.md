# Introduction

## async_api

该代码主要描述了cuda的异步特性，记录了gpu运行时间和cpu等待时间，其中，gpu运行时间通过`cudaEventElapsedTime`来统计。

注意，这里必须要在一个stream流上进行计算和统计。

## clock

通过reduce的方式来统计每个block的时间，最后计算每个block上花费的平均时间。

这里`clock_t`可以作为参数传入函数。
