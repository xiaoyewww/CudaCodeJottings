set(PROFILER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/profiler.cc)

add_library(profiler STATIC ${PROFILER_SRC})

target_include_directories(profiler PUBLIC ${PROJECT_SOURCE_DIR}/include)

