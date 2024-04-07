add_executable(cpp_overload cpp_overload.cu)

target_include_directories(cpp_overload PUBLIC ${PROJECT_SOURCE_DIR}/include/common ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME cpp_overload COMMAND $<TARGET_FILE:cpp_overload>)
