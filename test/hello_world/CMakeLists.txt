add_executable(hello_world hello_world.cu)

target_include_directories(hello_world PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(hello_world profiler)

add_test(NAME hello_world COMMAND $<TARGET_FILE:hello_world>)
