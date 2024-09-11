function(exe_build_func test_sample)
    add_executable(${test_sample} ${test_sample}.cu)
    target_include_directories(${test_sample} PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${test_sample} PRIVATE jotting-sdk profiler)
    add_test(NAME ${test_sample} COMMAND $<TARGET_FILE:${test_sample}>)
endfunction()


set(TEST_SAMPLES test_add test_softmax)
foreach(test_sample ${TEST_SAMPLES})
    exe_build_func(${test_sample})
endforeach()
