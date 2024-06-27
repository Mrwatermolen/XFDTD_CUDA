aux_source_directory(. XFDTD_CALCULATION_PARAM_CUDA_SRC)
add_library(xfdtd_calculation_param_cuda OBJECT ${XFDTD_CALCULATION_PARAM_CUDA_SRC})
set(${ALL_OBJECT_FILES} ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:xfdtd_calculation_param_cuda> PARENT_SCOPE)
