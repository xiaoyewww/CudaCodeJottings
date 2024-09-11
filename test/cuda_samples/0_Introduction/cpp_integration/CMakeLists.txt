add_executable(cpp_interation main.cpp cpp_integration.cu cpp_integration_gold.cpp)

target_include_directories(cpp_interation PUBLIC ${PROJECT_SOURCE_DIR}/include/common ${CUDAToolkit_INCLUDE_DIRS})

add_test(NAME cpp_interation COMMAND $<TARGET_FILE:cpp_interation>)
