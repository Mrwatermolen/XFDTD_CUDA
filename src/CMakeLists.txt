set(XFDTD_CUDA_PRIVATE_INLCUDES ${CMAKE_CURRENT_SOURCE_DIR}/include)
message(STATUS "${XFDTD_CUDA_PRIVATE_INLCUDES}")
include_directories(${XFDTD_CUDA_PRIVATE_INLCUDES})

add_subdirectory(boundary)
add_subdirectory(grid_space)
add_subdirectory(calculation_param)
add_subdirectory(domain)
add_subdirectory(material)
add_subdirectory(monitor)
add_subdirectory(nf2ff)
add_subdirectory(simulation)
add_subdirectory(updator)
add_subdirectory(waveform_source)

add_library(xfdtd_cuda ${ALL_OBJECT_FILES})

set(XFDTD_CUDA_LIBS 
  xfdtd_cuda_boundary
  xfdtd_cuda_grid_space xfdtd_cuda_calculation_param xfdtd_cuda_domain xfdtd_cuda_monitor
  xfdtd_cuda_material
  xfdtd_cuda_nf2ff
  xfdtd_cuda_simulation xfdtd_cuda_tfsf
  xfdtd_cuda_basic_updator
  xfdtd_cuda_ade_updator)

target_link_libraries(xfdtd_cuda PUBLIC xfdtd::xfdtd_core ${XFDTD_CUDA_LIBS})

target_include_directories(
  xfdtd_cuda PUBLIC $<BUILD_INTERFACE:${XFDTD_CUDA_PUBLIC_INCLUDES}>
                    $<INSTALL_INTERFACE:include>)

target_include_directories(
  xfdtd_cuda PRIVATE ${XFDTD_CUDA_PRIVATE_INCLUDE_DIR})

install(
  TARGETS xfdtd_cuda ${XFDTD_CUDA_LIBS}
  EXPORT xfdtd_cuda-targets
  ARCHIVE DESTINATION ${XFDTD_CUDA_INSTALL_PREFIX}/lib
  LIBRARY DESTINATION ${XFDTD_CUDA_INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${XFDTD_CUDA_INSTALL_PREFIX}/bin
  INCLUDES
  DESTINATION ${XFDTD_CUDA_INSTALL_PREFIX}/include)

install(DIRECTORY ${XFDTD_CUDA_PUBLIC_INCLUDES}/xfdtd_cuda
        DESTINATION ${XFDTD_CUDA_INSTALL_PREFIX}/include)
