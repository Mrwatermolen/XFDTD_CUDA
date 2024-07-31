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

#include <filesystem>
#include <iostream>
#include <xtensor/xnpy.hpp>

#include "xfdtd_cuda/simulation/simulation_hd.cuh"

auto benchmarkOnlyUpdator(dim3 grid_dim, dim3 block_dim) {
  constexpr double dl{2.5e-3};

  auto domain{std::make_shared<xfdtd::Object>(
      "domain",
      std::make_unique<xfdtd::Cube>(xfdtd::Vector{-0.175, -0.175, -0.175},
                                    xfdtd::Vector{0.35, 0.35, 0.35}),
      xfdtd::Material::createAir())};
  std::cout << "Size: " << 0.35 / dl << "\n";

  auto s{xfdtd::Simulation{dl, dl, dl, 0.9, xfdtd::ThreadConfig{1, 1, 1}}};
  s.addObject(domain);

  auto s_hd = xfdtd::cuda::SimulationHD{&s};
  s_hd.setGridDim(grid_dim);
  s_hd.setBlockDim(block_dim);
  std::chrono::high_resolution_clock::time_point start_time =
      std::chrono::high_resolution_clock::now();
  s_hd.run(2400);

  std::chrono::high_resolution_clock::time_point end_time =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> time_span = end_time - start_time;
  std::cout << "Elapsed time: " << time_span.count() << " ms\n";
  std::cout << "Elapsed time: " << time_span.count() / 1000 << " s\n";
}

auto benchmarkUpdaterAndTFSF(dim3 grid_dim, dim3 block_dim) {
  constexpr double dl{2.5e-3};
  using namespace std::string_view_literals;

  auto domain{std::make_shared<xfdtd::Object>(
      "domain",
      std::make_unique<xfdtd::Cube>(xfdtd::Vector{-0.175, -0.175, -0.175},
                                    xfdtd::Vector{0.35, 0.35, 0.35}),
      xfdtd::Material::createAir())};
  std::cout << "Size: " << 0.35 / dl << "\n";

  auto s{xfdtd::Simulation{dl, dl, dl, 0.9, xfdtd::ThreadConfig{1, 1, 1}}};
  s.addObject(domain);

  constexpr auto l_min{dl * 20};
  // constexpr auto f_max{3e8 / l_min};  // max frequency: 5 GHz in dl = 3e-3
  constexpr auto tau{l_min / 6e8};
  constexpr auto t_0{4.5 * tau};
  constexpr std::size_t tfsf_start{static_cast<std::size_t>(15)};
  auto tfsf{
      std::make_shared<xfdtd::TFSF3D>(tfsf_start, tfsf_start, tfsf_start, 0, 0,
                                      0, xfdtd::Waveform::gaussian(tau, t_0))};
  s.addWaveformSource(tfsf);

  auto s_hd = xfdtd::cuda::SimulationHD{&s};
  s_hd.setGridDim(grid_dim);
  s_hd.setBlockDim(block_dim);
  std::chrono::high_resolution_clock::time_point start_time =
      std::chrono::high_resolution_clock::now();
  s_hd.run(2400);

  std::chrono::high_resolution_clock::time_point end_time =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> time_span = end_time - start_time;
  std::cout << "Elapsed time: " << time_span.count() << " ms\n";
  std::cout << "Elapsed time: " << time_span.count() / 1000 << " s\n";
  std::cout << "Total number of grid: "
            << (0.35 / dl) * (0.35 / dl) * (0.35 / dl) << "\n";
  std::cout << "Per second for grid: "
            << 2400 * (0.35 / dl) * (0.35 / dl) * (0.35 / dl) /
                   (time_span.count() / 1000)
            << "\n";
}

auto benchmarkUpdatorAndTFSFAndPML(dim3 grid_dim, dim3 block_dim) {
  std::cout << "Benchmark for Updator and TFSF and PML\n";

  constexpr double dl{2.5e-3};

  auto domain{std::make_shared<xfdtd::Object>(
      "domain",
      std::make_unique<xfdtd::Cube>(xfdtd::Vector{-0.175, -0.175, -0.175},
                                    xfdtd::Vector{0.35, 0.35, 0.35}),
      xfdtd::Material::createAir())};
  std::cout << "Size: " << 0.35 / dl << "\n";

  auto s{xfdtd::Simulation{dl, dl, dl, 0.9, xfdtd::ThreadConfig{1, 1, 1}}};
  s.addObject(domain);

  constexpr auto l_min{dl * 20};
  // constexpr auto f_max{3e8 / l_min};  // max frequency: 5 GHz in dl = 3e-3
  constexpr auto tau{l_min / 6e8};
  constexpr auto t_0{4.5 * tau};
  constexpr std::size_t tfsf_start{static_cast<std::size_t>(15)};
  auto tfsf{
      std::make_shared<xfdtd::TFSF3D>(tfsf_start, tfsf_start, tfsf_start, 0, 0,
                                      0, xfdtd::Waveform::gaussian(tau, t_0))};
  s.addWaveformSource(tfsf);

  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XP));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YP));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZP));

  auto s_hd = xfdtd::cuda::SimulationHD{&s};
  s_hd.setGridDim(grid_dim);
  s_hd.setBlockDim(block_dim);
  std::chrono::high_resolution_clock::time_point start_time =
      std::chrono::high_resolution_clock::now();
  s_hd.run(2400);

  std::chrono::high_resolution_clock::time_point end_time =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> time_span = end_time - start_time;
  std::cout << "Elapsed time: " << time_span.count() << " ms\n";
  std::cout << "Elapsed time: " << time_span.count() / 1000 << " s\n";
  std::cout << "Total number of grid: "
            << (0.35 / dl) * (0.35 / dl) * (0.35 / dl) << "\n";
  std::cout << "Per second for grid: "
            << 2400 * (0.35 / dl) * (0.35 / dl) * (0.35 / dl) /
                   (time_span.count() / 1000)
            << "\n";
}

auto benchmarkUpdatorAndTFSFAndPMLAndNF2FF(dim3 grid_dim, dim3 block_dim) {
  std::cout << "Benchmark for Updator and TFSF and PML and NF2FF\n";

  constexpr double dl{2.5e-3};
  using namespace std::string_view_literals;
  constexpr auto data_path_str = "./tmp/dielectric_sphere_scatter"sv;
  const auto data_path = std::filesystem::path{data_path_str};

  auto domain{std::make_shared<xfdtd::Object>(
      "domain",
      std::make_unique<xfdtd::Cube>(xfdtd::Vector{-0.175, -0.175, -0.175},
                                    xfdtd::Vector{0.35, 0.35, 0.35}),
      xfdtd::Material::createAir())};
  std::cout << "Size: " << 0.35 / dl << "\n";

  auto dielectric_sphere{std::make_shared<xfdtd::Object>(
      "dielectric_sphere",
      std::make_unique<xfdtd::Sphere>(xfdtd::Vector{0, 0, 0}, 0.1),
      std::make_unique<xfdtd::Material>(
          "a", xfdtd::ElectroMagneticProperty{3, 2, 0, 0}))};

  auto s{xfdtd::Simulation{dl, dl, dl, 0.9, xfdtd::ThreadConfig{1, 1, 1}}};
  s.addObject(domain);
  s.addObject(dielectric_sphere);

  constexpr auto l_min{dl * 20};
  // constexpr auto f_max{3e8 / l_min};  // max frequency: 5 GHz in dl = 3e-3
  constexpr auto tau{l_min / 6e8};
  constexpr auto t_0{4.5 * tau};
  constexpr std::size_t tfsf_start{static_cast<std::size_t>(15)};
  auto tfsf{
      std::make_shared<xfdtd::TFSF3D>(tfsf_start, tfsf_start, tfsf_start, 0, 0,
                                      0, xfdtd::Waveform::gaussian(tau, t_0))};
  s.addWaveformSource(tfsf);

  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XP));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YP));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZP));

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

  nffft_fd->processFarField(
      xt::linspace<double>(-xfdtd::constant::PI, xfdtd::constant::PI, 360), 0,
      "xz");

  auto time = tfsf->waveform()->time();
  auto incident_wave_data = tfsf->waveform()->value();
  xt::dump_npy((data_path / "time.npy").string(), time);
  xt::dump_npy((data_path / "incident_wave.npy").string(), incident_wave_data);

  std::chrono::high_resolution_clock::time_point end_time =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> time_span = end_time - start_time;
  std::cout << "Elapsed time: " << time_span.count() << " ms\n";
  std::cout << "Elapsed time: " << time_span.count() / 1000 << " s\n";
  std::cout << "Total number of grid: "
            << (0.35 / dl) * (0.35 / dl) * (0.35 / dl) << "\n";
  std::cout << "Per second for grid: "
            << 2400 * (0.35 / dl) * (0.35 / dl) * (0.35 / dl) /
                   (time_span.count() / 1000)
            << "\n";
}

int main(int argc, char** argv) {
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

  // benchmarkOnlyUpdator(grid_dim, block_dim);
  // benchmarkUpdaterAndTFSF(grid_dim, block_dim);
  // benchmarkUpdatorAndTFSFAndPML(grid_dim, block_dim);
  benchmarkUpdatorAndTFSFAndPMLAndNF2FF(grid_dim, block_dim);
  return 0;
}
