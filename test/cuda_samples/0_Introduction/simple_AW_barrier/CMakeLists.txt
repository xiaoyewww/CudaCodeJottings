add_executable(simple_AW_barrier simple_AW_barrier.cu)

target_include_directories(simple_AW_barrier PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(simple_AW_barrier PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

target_link_libraries(simple_AW_barrier
    PUBLIC
        ${CUDA_cublas_LIBRARY} ${CUDA_cusparse_LIBRARY})

set_property(TARGET simple_AW_barrier PROPERTY CUDA_STANDARD 17)

add_test(NAME simple_AW_barrier COMMAND $<TARGET_FILE:simple_AW_barrier>)
