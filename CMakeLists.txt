cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDA_CODE_JOTTINGS LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

option(WITH_TEST "ctest" ON)

enable_language(CUDA)

if(WITH_TEST)
    enable_testing()
    add_subdirectory(test)
endif()
