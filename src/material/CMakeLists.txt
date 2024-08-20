aux_source_directory(. XFDTD_CUDA_MATERIAL_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ade_method XFDTD_CUDA_ADE_METHOD_SRC)
add_library(xfdtd_cuda_material OBJECT ${XFDTD_CUDA_MATERIAL_SRC} ${XFDTD_CUDA_ADE_METHOD_SRC})

target_include_directories(xfdtd_cuda_material PRIVATE ${XFDTD_CUDA_PRIVATE_INLCUDES})
set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_material> PARENT_SCOPE)
