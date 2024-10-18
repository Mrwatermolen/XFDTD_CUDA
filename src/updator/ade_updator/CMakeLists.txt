aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} XFDTD_CUDA_ADE_UPDATOR_SRC)
add_library(xfdtd_cuda_ade_updator OBJECT ${XFDTD_CUDA_ADE_UPDATOR_SRC})

set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_ade_updator> PARENT_SCOPE)