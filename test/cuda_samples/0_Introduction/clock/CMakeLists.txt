add_executable(clock clock.cu)

target_include_directories(clock PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

add_test(NAME clock COMMAND $<TARGET_FILE:clock>)
