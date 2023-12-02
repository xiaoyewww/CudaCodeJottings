add_executable(hello_world hello_world.cu)

add_test(NAME hello_world COMMAND $<TARGET_FILE:hello_world>)
