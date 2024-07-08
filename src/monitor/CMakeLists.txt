aux_source_directory(. XFDTD_CUDA_MONITOR_SRC)
add_library(xfdtd_cuda_monitor OBJECT ${XFDTD_CUDA_MONITOR_SRC})

target_include_directories(xfdtd_cuda_monitor PRIVATE ${XFDTD_CUDA_PRIVATE_INLCUDES})
set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_monitor> PARENT_SCOPE)