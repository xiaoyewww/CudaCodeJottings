add_executable(add_test add_test.cu)

target_include_directories(add_test PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(add_test PRIVATE jotting-sdk profiler)

add_test(NAME add_test COMMAND $<TARGET_FILE:add_test>)
