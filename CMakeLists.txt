cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDA_CODE_JOTTINGS LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

option(WITH_TEST "whether to build with unittest" ON)
option(WITH_PROFILER "whether to build with profiler" ON)

enable_language(CUDA)

if(WITH_PROFILER)
    add_definitions(-DWITH_PROFILER)
endif()

add_subdirectory(src)

if(WITH_TEST)
    enable_testing()
    add_subdirectory(test)
endif()
