aux_source_directory(. XFDTD_CUDA_SIMULATION_SRC)
add_library(xfdtd_cuda_simulation OBJECT ${XFDTD_CUDA_SIMULATION_SRC})
set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_simulation> PARENT_SCOPE)
