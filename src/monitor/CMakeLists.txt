aux_source_directory(. XFDTD_CUDA_MONITOR_SRC)
add_library(xfdtd_cuda_monitor OBJECT ${XFDTD_CUDA_MONITOR_SRC})

set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_monitor> PARENT_SCOPE)