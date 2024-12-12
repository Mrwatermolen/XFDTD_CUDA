cmake_minimum_required(VERSION 3.25.0)

project(XFDTD_CUDA VERSION 0.0.0 LANGUAGES CXX CUDA)

set(XFDTD_CUDA_MAIN_PROJECT OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(XFDTD_CUDA_MAIN_PROJECT ON)
endif()

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed")
endif()

# if not set, set default build type to Release
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set cuda compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(XFDTD_CUDA_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  add_compile_definitions(XFDTD_CUDA_CXX_MSVC)
endif()

find_package(xfdtd_core REQUIRED)

set(XFDTD_CUDA_PUBLIC_INCLUDES ${PROJECT_SOURCE_DIR}/include)
include_directories(${XFDTD_CUDA_PUBLIC_INCLUDES})
link_libraries(xfdtd::xfdtd_core)

add_subdirectory(src)
add_subdirectory(test)

install(
  EXPORT xfdtd_cuda-targets
  FILE xfdtd_cuda-targets.cmake
  NAMESPACE xfdtd::
  DESTINATION ${XFDTD_CUDA_INSTALL_PREFIX}/lib/cmake/xfdtd_cuda)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  cmake/xfdtd_cuda-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/xfdtd_cuda-config.cmake
  INSTALL_DESTINATION ${XFDTD_CUDA_INSTALL_PREFIX}/lib/cmake/xfdtd_cuda)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/xfdtd_cuda-config_version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xfdtd_cuda-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/xfdtd_cuda-config_version.cmake
        DESTINATION ${XFDTD_CUDA_INSTALL_PREFIX}/lib/cmake/xfdtd_cuda)