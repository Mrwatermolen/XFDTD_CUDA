#include <xfdtd/boundary/pml.h>
#include <xfdtd/common/constant.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/material/material.h>
#include <xfdtd/monitor/field_monitor.h>
#include <xfdtd/monitor/movie_monitor.h>
#include <xfdtd/nffft/nffft_frequency_domain.h>
#include <xfdtd/object/object.h>
#include <xfdtd/shape/sphere.h>
#include <xfdtd/simulation/simulation.h>
#include <xfdtd/waveform_source/tfsf_3d.h>

#include <filesystem>
#include <iostream>

#include "xfdtd_cuda/simulation/simulation_hd.cuh"

auto dielectricSphereScatter(dim3 grid_dim, dim3 block_dim) -> void {
  constexpr double dl{5e-3};
  using namespace std::string_view_literals;
  constexpr auto data_path_str = "./tmp/dielectric_sphere_scatter"sv;
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
  std::chrono::high_resolution_clock::time_point end_time =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> time_span = end_time - start_time;
  std::cout << "Elapsed time: " << time_span.count() << " ms\n";
  std::cout << "Elapsed time: " << time_span.count() / 1000 << " s\n";
}

auto main(int argc, char** argv) -> int {
  dim3 grid_dim{4, 4, 4};
  dim3 block_dim{8, 8, 8};

  if (7 <= argc) {
    grid_dim.x = std::atoi(argv[1]);
    grid_dim.y = std::atoi(argv[2]);
    grid_dim.z = std::atoi(argv[3]);
    block_dim.x = std::atoi(argv[4]);
    block_dim.y = std::atoi(argv[5]);
    block_dim.z = std::atoi(argv[6]);
  }

  std::cout << "Grid size: (" << grid_dim.x << ", " << grid_dim.y << ", "
            << grid_dim.z << "), Block size: (" << block_dim.x << ", "
            << block_dim.y << ", " << block_dim.z << ")\n";

  dielectricSphereScatter(grid_dim, block_dim);
  return 0;
}
