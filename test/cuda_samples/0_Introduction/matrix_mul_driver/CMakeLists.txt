# not determine which sm arch.
add_custom_command(
    OUTPUT matrixMul_kernel64.fatbin
    COMMAND ${CUDA_NVCC_EXECUTABLE} -fatbin -o matrixMul_kernel64.fatbin ${CMAKE_CURRENT_SOURCE_DIR}/matrixMul_kernel.cu
    DEPENDS matrixMul_kernel.cu
    COMMENT "Generating fatbin file"
)
add_custom_target(dummy ALL DEPENDS matrixMul_kernel64.fatbin)

add_executable(matrix_mul_driver matrix_mul_driver.cpp)

# if no add, cannot find symbols.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lcuda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcuda")

target_include_directories(matrix_mul_driver PUBLIC ${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
target_include_directories(matrix_mul_driver PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

target_link_libraries(matrix_mul_driver
    PUBLIC
        ${CUDA_cudadevrt_LIBRARY} OpenMP::OpenMP_CXX)

add_test(NAME matrix_mul_driver COMMAND $<TARGET_FILE:matrix_mul_driver>)
