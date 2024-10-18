aux_source_directory(. XFDTD_CUDA_DOMAIN_SRC)
add_library(xfdtd_cuda_domain OBJECT ${XFDTD_CUDA_DOMAIN_SRC})

set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_domain> PARENT_SCOPE)