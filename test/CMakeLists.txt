add_executable(cuda_fdtd_2d 2d.cu)
target_link_libraries(cuda_fdtd_2d PRIVATE xfdtd::xfdtd_core)

add_executable(cylinder_scatter_2d cylinder_scatter_2d.cu)
target_link_libraries(cylinder_scatter_2d PRIVATE xfdtd_cuda)

add_executable(dielectric_sphere_scatter dielectric_sphere_scatter.cu)
target_link_libraries(dielectric_sphere_scatter PRIVATE xfdtd_cuda)

add_executable(device_query device_query.cu)

add_executable(benchmark benchmark.cu)
target_link_libraries(benchmark PRIVATE xfdtd_cuda)
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
