aux_source_directory(. XFDTD_CALCULATION_PARAM_CUDA_SRC)
add_library(xfdtd_cuda_calculation_param OBJECT ${XFDTD_CALCULATION_PARAM_CUDA_SRC})
set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_cuda_calculation_param> PARENT_SCOPE)
