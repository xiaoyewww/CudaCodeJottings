#include "sys/time.h"

#ifdef WITH_PROFILER

#define PROFILER_FUNC(func, args...)                                         \
  {                                                                          \
    printf("=============== The profiler is beginning ===============\n");   \
    auto begin_time = CpuSecond();                                           \
    func(##args);                                                            \
    auto end_time = CpuSecond();                                             \
    printf("=============== The profiler is done ====================\n");   \
    printf("The time of %s is %f seconds.\n", #func, end_time - begin_time); \
  }

#define PROFILER_FUNC_TIMES(func, times, args...)                          \
  {                                                                        \
    printf("=============== The profiler is beginning ===============\n"); \
    auto begin_time = CpuSecond();                                         \
    for (int i = 0; i < times; ++i) {                                      \
      func(##args);                                                        \
    }                                                                      \
    auto end_time = CpuSecond();                                           \
    printf("=============== The profiler is done ====================\n"); \
    auto total_time = end_time - begin_time;                               \
    printf("The total time of %s is %f seconds, average time is %f.\n",    \
           #func,                                                          \
           total_time,                                                     \
           total_time / times);                                            \
  }

#define PROFILER_FUNC_10TIMES(func, args...) \
  PROFILER_FUNC_TIMES(func, 10, ##args)
#define PROFILER_FUNC_100TIMES(func, args...) \
  PROFILER_FUNC_TIMES(func, 100, ##args)
#define PROFILER_FUNC_1000TIMES(func, args...) \
  PROFILER_FUNC_TIMES(func, 1000, ##args)

#define PROFILER_CUDA_FUNC(func, blk, idx, args...)            \
  {                                                            \
    printf(                                                    \
        "=============== The cuda func profiler is beginning " \
        "===============\n");                                  \
    auto cuda_begin_time = CpuSecond();                        \
    func<<<blk, idx>>>(##args);                                \
    cudaDeviceSynchronize();                                   \
    auto cuda_end_time = CpuSecond();                          \
    printf(                                                    \
        "=============== The cuda func profiler is done "      \
        "====================\n");                             \
    printf("The time of %s is %f seconds.\n",                  \
           #func,                                              \
           cuda_end_time - cuda_begin_time);                   \
  }

#define PROFILER_CUDA_FUNC_TIMES(func, blk, idx, times, args...)             \
  {                                                                          \
    printf(                                                                  \
        "=============== The cuda func profiler is beginning "               \
        "===============\n");                                                \
    auto cuda_begin_time = CpuSecond();                                      \
    for (int i = 0; i < times; ++i) {                                        \
      func<<<blk, idx>>>(##args);                                            \
      cudaDeviceSynchronize();                                               \
    }                                                                        \
    auto cuda_end_time = CpuSecond();                                        \
    printf(                                                                  \
        "=============== The cuda func profiler is done "                    \
        "====================\n");                                           \
    auto cuda_total_time = cuda_end_time - cuda_begin_time;                  \
    printf(                                                                  \
        "The total time of %s is %f seconds, average time is %f seconds.\n", \
        #func,                                                               \
        cuda_total_time,                                                     \
        cuda_total_time / times);                                            \
  }

#define PROFILER_CUDA_FUNC_10TIMES(func, blk, idx, args...) \
  PROFILER_CUDA_FUNC_TIMES(func, blk, idx, 10, ##args)
#define PROFILER_CUDA_FUNC_100TIMES(func, blk, idx, args...) \
  PROFILER_CUDA_FUNC_TIMES(func, blk, idx, 100, ##args)
#define PROFILER_CUDA_FUNC_1000TIMES(func, blk, idx, args...) \
  PROFILER_CUDA_FUNC_TIMES(func, blk, idx, 1000, ##args)

#else

#define PROFILER_FUNC(func, args...) func(##args)
#define PROFILER_FUNC_TIMES(func, times, args...) func(##args)
#define PROFILER_FUNC_10TIMES(func, args...) func(##args)
#define PROFILER_FUNC_100TIMES(func, args...) func(##args)
#define PROFILER_FUNC_1000TIMES(func, args...) func(##args)

#define PROFILER_CUDA_FUNC(func, blk, idx, args...) func<<<blk, idx>>>(##args)
#define PROFILER_CUDA_FUNC_TIMES(func, blk, idx, times, args...) \
  func<<<blk, idx>>>(##args)
#define PROFILER_CUDA_FUNC_10TIMES(func, blk, idx, args...) \
  func<<<blk, idx>>>(##args)
#define PROFILER_CUDA_FUNC_100TIMES(func, blk, idx, args...) \
  func<<<blk, idx>>>(##args)
#define PROFILER_CUDA_FUNC_1000TIMES(func, blk, idx, args...) \
  func<<<blk, idx>>>(##args)

#endif

double CpuSecond();
