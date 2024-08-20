aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} XFDTD_CUDA_UPDATOR_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ade_updator XFDTD_CUDA_UPDATOR_SRC)
add_library(xfdtd_cuda_updator OBJECT ${XFDTD_CUDA_UPDATOR_SRC})

target_include_directories(xfdtd_cuda_updator PRIVATE ${XFDTD_CUDA_PRIVATE_INLCUDES})
set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_updator> PARENT_SCOPE)