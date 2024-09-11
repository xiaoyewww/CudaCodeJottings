add_subdirectory(profiler)

set(KERNEL_SRC  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/add.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/kernel/softmax.cpp
)

set_source_files_properties(${KERNEL_SRC} PROPERTIES LANGUAGE CUDA)

add_library(jotting-sdk SHARED ${KERNEL_SRC})

# set_target_properties(jotting-sdk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(jotting-sdk
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    jotting-sdk
    profiler
)
