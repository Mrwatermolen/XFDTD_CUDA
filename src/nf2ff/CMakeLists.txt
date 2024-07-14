aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/frequency_domain XFDTD_CUDA_FREQUENCY_DOAMIN_NF2FF_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/time_domain XFDTD_CUDA_TIME_DOMAIN_NF2FF_SRC)
set(XFDTD_CUDA_NF2FF_SRC ${XFDTD_CUDA_FREQUENCY_DOAMIN_NF2FF_SRC} ${XFDTD_CUDA_TIME_DOMAIN_NF2FF_SRC})
# message("Adding XFDTD_CUDA_NF2FF_SRC files: ${XFDTD_CUDA_NF2FF_SRC}")
add_library(xfdtd_cuda_nf2ff OBJECT ${XFDTD_CUDA_NF2FF_SRC})

target_include_directories(xfdtd_cuda_nf2ff PRIVATE ${XFDTD_CUDA_PRIVATE_INLCUDES})
target_link_libraries(xfdtd_cuda_nf2ff PUBLIC xfdtd::xfdtd_core)
set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_nf2ff> PARENT_SCOPE)