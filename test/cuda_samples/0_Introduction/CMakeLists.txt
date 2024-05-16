add_subdirectory(unified_memory_streams)
add_subdirectory(async_api)
add_subdirectory(cpp11_cuda)
add_subdirectory(clock)
add_subdirectory(concurrent_kernels)
add_subdirectory(cpp_integration)
add_subdirectory(cpp_overload)
add_subdirectory(cuda_openmp)
add_subdirectory(fp16_scalar_product)
add_subdirectory(matrix_mul)
add_subdirectory(matrix_mul_driver)
add_subdirectory(simple_assert)

# the sample of cannot compile in my env, casue by:
# https://forums.developer.nvidia.com/t/cuda-12-1-error-when-building-cuda-samples/246465
# add_subdirectory(simple_AW_barrier)
