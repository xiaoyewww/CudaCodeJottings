add_executable(unified_memory_streams unified_memory_streams.cu)

target_include_directories(unified_memory_streams PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(unified_memory_streams PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

target_link_libraries(unified_memory_streams
    PRIVATE
        jotting-sdk profiler
    PUBLIC
        ${CUDA_cublas_LIBRARY} ${CUDA_cusparse_LIBRARY} OpenMP::OpenMP_CXX)

add_test(NAME unified_memory_streams COMMAND $<TARGET_FILE:unified_memory_streams>)
