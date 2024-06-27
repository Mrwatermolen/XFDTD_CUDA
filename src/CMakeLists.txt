add_subdirectory(grid_space)

message(STATUS "CUDA_INCLUDE_DIRS: ${ALL_OBJECT_FILES}")
add_library(xfdtd_cuda ${ALL_OBJECT_FILES})

set(XFDTD_CUDA_LIBS xfdtd_cuda_grid_space
)

target_link_libraries(xfdtd_cuda PUBLIC xfdtd::xfdtd_core ${XFDTD_CUDA_LIBS})
