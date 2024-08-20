aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} XFDTD_CUDA_BASIC_UPDATOR_SRC)
add_library(xfdtd_cuda_basic_updator OBJECT ${XFDTD_CUDA_BASIC_UPDATOR_SRC})

target_include_directories(xfdtd_cuda_basic_updator PRIVATE ${XFDTD_CUDA_PRIVATE_INLCUDES})
set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_basic_updator> PARENT_SCOPE)