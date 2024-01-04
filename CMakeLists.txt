cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDA_CODE_JOTTINGS LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

option(WITH_TEST "whether to build with unittest" ON)
option(WITH_PROFILER "whether to build with profiler" ON)

enable_language(CUDA)
find_package(CUDA)

message(INFO "CUDA_cublas_LIBRARY: ${CUDA_cublas_LIBRARY}")
message(INFO "CUDA_cusparse_LIBRARY: ${CUDA_cusparse_LIBRARY}")

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # specify for cuda
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found!")
endif()

if(WITH_PROFILER)
    add_definitions(-DWITH_PROFILER)
endif()

add_subdirectory(src)

if(WITH_TEST)
    add_subdirectory(test)
endif()
