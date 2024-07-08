aux_source_directory(. XFDTD_CUDA_SIMULATION_SRC)
add_library(xfdtd_cuda_simulation OBJECT ${XFDTD_CUDA_SIMULATION_SRC})

target_include_directories(xfdtd_cuda_simulation PRIVATE ${XFDTD_CUDA_PRIVATE_INLCUDES})
target_link_libraries(xfdtd_cuda_simulation PUBLIC xfdtd::xfdtd_core)
set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_simulation> PARENT_SCOPE)