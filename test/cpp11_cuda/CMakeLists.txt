add_executable(cpp11_cuda cpp11_cuda.cu)

target_include_directories(cpp11_cuda PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

add_test(NAME cpp11_cuda COMMAND $<TARGET_FILE:cpp11_cuda>)
