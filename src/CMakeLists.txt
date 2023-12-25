add_subdirectory(profiler)

set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernel/add.cu)

add_library(jotting-sdk SHARED ${KERNEL_SRC})

target_link_libraries(
    jotting-sdk PRIVATE
    profiler
)
