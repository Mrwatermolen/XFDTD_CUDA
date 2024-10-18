aux_source_directory(. XFDTD_CUDA_BOUNDARY_SRC)
add_library(xfdtd_cuda_boundary OBJECT ${XFDTD_CUDA_BOUNDARY_SRC})
set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_boundary> PARENT_SCOPE)
