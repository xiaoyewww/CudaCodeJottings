add_executable(concurrent_kernels concurrent_kernels.cu)

target_include_directories(concurrent_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

add_test(NAME concurrent_kernels COMMAND $<TARGET_FILE:concurrent_kernels>)
