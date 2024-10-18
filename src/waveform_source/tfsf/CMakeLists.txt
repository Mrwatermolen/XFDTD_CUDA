aux_source_directory(. XFDTD_CUDA_TFSF_SRC)
add_library(xfdtd_cuda_tfsf OBJECT ${XFDTD_CUDA_TFSF_SRC})

set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_tfsf> PARENT_SCOPE)