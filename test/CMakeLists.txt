add_executable(cuda_fdtd_2d 2d.cu)
target_link_libraries(cuda_fdtd_2d PRIVATE xfdtd::xfdtd_core)

add_executable(main main.cu)
target_link_libraries(main PRIVATE xfdtd_cuda)

add_executable(device_query device_query.cu)