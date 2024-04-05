add_executable(matrix_mul matrix_mul.cu)

target_include_directories(matrix_mul PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(matrix_mul PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

add_test(NAME matrix_mul COMMAND $<TARGET_FILE:matrix_mul>)
