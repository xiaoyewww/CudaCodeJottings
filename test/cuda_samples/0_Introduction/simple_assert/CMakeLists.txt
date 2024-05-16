add_executable(simple_assert simple_assert.cu)

target_include_directories(simple_assert PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(simple_assert PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

target_link_libraries(simple_assert
    PUBLIC
        ${CUDA_cublas_LIBRARY} ${CUDA_cusparse_LIBRARY})

set_property(TARGET simple_assert PROPERTY CUDA_STANDARD 17)

add_test(NAME simple_assert COMMAND $<TARGET_FILE:simple_assert>)
