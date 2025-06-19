# XFDTD_CUDA

This is a part of the XFDTD project. It supports CUDA acceleration for the FDTD algorithm.

## Getting Started

You will require the following libraries to build the project: [XFDTD_CORE](https://github.com/Mrwatermolen/XFDTD_CORE)

### Install from source

```bash
cmake -S . -B build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build -t install
```

### Use in your project

Here is a example to simulate a dielectric sphere scatter a plane wave.

Assuming you have installed the library in /path/to/install, you can use the following CMakeLists.txt to use the library in your project. The folder structure:

```tree
.
├── CMakeLists.txt
└── main.cu
```

The `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.25)

project(PROJECT_NAME VERSION 0.0.0 LANGUAGES CXX CUDA)

set(PROJECT_NAME_MAIN_PROJECT OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(PROJECT_NAME_MAIN_PROJECT ON)
endif()

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(xfdtd_core REQUIRED)
find_package(xfdtd_cuda REQUIRED)

add_executable(${PROJECT_NAME} main.cu)
target_link_libraries(${PROJECT_NAME} PRIVATE xfdtd::xfdtd_core xfdtd::xfdtd_cuda)
```

The `main.cu`:

```cpp
#include <filesystem>
#include <iostream>
#include <xfdtd/boundary/pml.h>
#include <xfdtd/common/constant.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/material/material.h>
#include <xfdtd/monitor/field_monitor.h>
#include <xfdtd/monitor/movie_monitor.h>
#include <xfdtd/nffft/nffft_frequency_domain.h>
#include <xfdtd/nffft/nffft_time_domain.h>
#include <xfdtd/object/object.h>
#include <xfdtd/shape/sphere.h>
#include <xfdtd/simulation/simulation.h>
#include <xfdtd/waveform_source/tfsf_3d.h>

#include <xfdtd_cuda/simulation/simulation_hd.cuh>

#include <xtensor/xnpy.hpp>

auto dielectricSphereScatter(dim3 grid_dim, dim3 block_dim) -> void {
  constexpr double dl{2.5e-3};
  using namespace std::string_view_literals;
  constexpr auto data_path_str = "./data/dielectric_sphere_scatter"sv;
  const auto data_path = std::filesystem::path{data_path_str};

  auto domain{std::make_shared<xfdtd::Object>(
      "domain",
      std::make_unique<xfdtd::Cube>(xfdtd::Vector{-0.175, -0.175, -0.175},
                                    xfdtd::Vector{0.35, 0.35, 0.35}),
      xfdtd::Material::createAir())};

  auto dielectric_sphere{std::make_shared<xfdtd::Object>(
      "dielectric_sphere",
      std::make_unique<xfdtd::Sphere>(xfdtd::Vector{0, 0, 0}, 0.1),
      std::make_unique<xfdtd::Material>(
          "a", xfdtd::ElectroMagneticProperty{3, 2, 0, 0}))};

  constexpr auto l_min{dl * 20};
  // constexpr auto f_max{3e8 / l_min};  // max frequency: 5 GHz in dl = 3e-3
  constexpr auto tau{l_min / 6e8};
  constexpr auto t_0{4.5 * tau};
  constexpr std::size_t tfsf_start{static_cast<size_t>(15)};
  auto tfsf{
      std::make_shared<xfdtd::TFSF3D>(tfsf_start, tfsf_start, tfsf_start, 0, 0,
                                      0, xfdtd::Waveform::gaussian(tau, t_0))};

  auto movie_ex_xz{std::make_shared<xfdtd::MovieMonitor>(
      std::make_unique<xfdtd::FieldMonitor>(
          std::make_unique<xfdtd::Cube>(xfdtd::Vector{-0.175, 0, -0.175},
                                        xfdtd::Vector{0.35, dl, 0.35}),
          xfdtd::EMF::Field::EX, "", ""),
      20, "movie_ex_xz", (data_path / "movie_ex_xz").string())};

  auto s{xfdtd::Simulation{dl, dl, dl, 0.9, xfdtd::ThreadConfig{1, 1, 1}}};
  s.addObject(domain);
  s.addObject(dielectric_sphere);
  s.addWaveformSource(tfsf);

  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XP));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YP));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZP));
  s.addMonitor(movie_ex_xz);

  constexpr std::size_t nffft_start{static_cast<size_t>(11)};
  auto nffft_fd{std::make_shared<xfdtd::NFFFTFrequencyDomain>(
      nffft_start, nffft_start, nffft_start, xt::xarray<double>{1e9},
      (data_path / "fd").string())};

  s.addNF2FF(nffft_fd);

  auto s_hd = xfdtd::cuda::SimulationHD{&s};
  s_hd.setGridDim(grid_dim);
  s_hd.setBlockDim(block_dim);
  std::chrono::high_resolution_clock::time_point start_time =
      std::chrono::high_resolution_clock::now();
  s_hd.run(2400);

  nffft_fd->setOutputDir((data_path / "fd").string());
  nffft_fd->processFarField(
      xt::linspace<double>(-xfdtd::constant::PI, xfdtd::constant::PI, 360), 0,
      "xz");
  nffft_fd->processFarField(
      xt::linspace<double>(-xfdtd::constant::PI, xfdtd::constant::PI, 360),
      xfdtd::constant::PI * 0.5, "yz");

  if (!s.isRoot()) {
    return;
  }

  auto time{tfsf->waveform()->time()};
  auto incident_wave_data{tfsf->waveform()->value()};
  xt::dump_npy((data_path / "time.npy").string(), time);
  xt::dump_npy((data_path / "incident_wave.npy").string(), incident_wave_data);

  std::chrono::high_resolution_clock::time_point end_time =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> time_span = end_time - start_time;
  std::cout << "Elapsed time: " << time_span.count() << " ms\n";
  std::cout << "Elapsed time: " << time_span.count() / 1000 << " s\n";
}

int main() {
  dielectricSphereScatter(dim3{64, 64, 1}, dim3{1, 1, 64});
  return 0;
}
```

build the project:

```bash
cmake -S . -B build
cmake --build build
```

run the project:

```bash
./build/PROJECT_NAME
```

## Supported Features

We are still working on the CUDA acceleration for the FDTD algorithm. The following features are supported:

* Perfectly Matched Layer (PML): ✅
* Movie Monitor: ✅
* Near to Far Field Transformation: Frequency Domain: ✅ Time Domain: ❌
* TFSF Source: ✅
* 3D Objects: ✅
* Dispersion medium model: MLorentz Model✅( including:  Drude Model, Debye Model, Lorentz Model, CCPR Model)

And more features are coming soon.
