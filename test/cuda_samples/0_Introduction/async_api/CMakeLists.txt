add_executable(async_api async_api.cu)

target_include_directories(async_api PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

add_test(NAME async_api COMMAND $<TARGET_FILE:async_api>)
