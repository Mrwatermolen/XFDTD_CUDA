aux_source_directory(. XFDTD_CUDA_GRID_SPACE_SRC)
add_library(xfdtd_cuda_grid_space OBJECT ${XFDTD_CUDA_GRID_SPACE_SRC})

set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_grid_space> PARENT_SCOPE)