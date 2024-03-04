add_executable(cuda_openmp cuda_openmp.cu)

target_include_directories(cuda_openmp PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

target_link_libraries(cuda_openmp PUBLIC OpenMP::OpenMP_CXX)

add_test(NAME cuda_openmp COMMAND $<TARGET_FILE:cuda_openmp>)
