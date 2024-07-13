set(XFDTD_CUDA_PRIVATE_INLCUDES ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(boundary)
add_subdirectory(grid_space)
add_subdirectory(calculation_param)
add_subdirectory(domain)
add_subdirectory(monitor)
add_subdirectory(nf2ff)
add_subdirectory(simulation)
add_subdirectory(updator)
add_subdirectory(waveform_source)

add_library(xfdtd_cuda ${ALL_OBJECT_FILES})

set(XFDTD_CUDA_LIBS 
  xfdtd_cuda_boundary
  xfdtd_cuda_grid_space xfdtd_cuda_calculation_param xfdtd_cuda_domain xfdtd_cuda_monitor
  xfdtd_cuda_nf2ff
  xfdtd_cuda_simulation xfdtd_cuda_tfsf xfdtd_cuda_updator)

target_link_libraries(xfdtd_cuda PUBLIC xfdtd::xfdtd_core ${XFDTD_CUDA_LIBS})
