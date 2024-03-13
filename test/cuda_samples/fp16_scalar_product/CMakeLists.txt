add_executable(fp16_scalar_product fp16_scalar_product.cu)

target_include_directories(fp16_scalar_product PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(fp16_scalar_product PUBLIC ${PROJECT_SOURCE_DIR}/include/common)

add_test(NAME fp16_scalar_product COMMAND $<TARGET_FILE:fp16_scalar_product>)
