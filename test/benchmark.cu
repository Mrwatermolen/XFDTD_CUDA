#include <xfdtd/boundary/pml.h>
#include <xfdtd/common/constant.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/material/material.h>
#include <xfdtd/nffft/nffft_frequency_domain.h>
#include <xfdtd/object/object.h>
#include <xfdtd/shape/sphere.h>
#include <xfdtd/simulation/simulation.h>
#include <xfdtd/waveform_source/tfsf_3d.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <xtensor/xnpy.hpp>

#include "argparse/argparse.hpp"
#include "xfdtd_cuda/simulation/simulation_hd.cuh"

static constexpr auto typeToString(int type) {
  switch (type) {
    case 0:
      return "Only Updator";
    case 1:
      return "Updator and TFSF";
    case 2:
      return "Updator and TFSF and PML";
    case 3:
      return "Updator and TFSF and PML and NF2FF";
    default:
      return "Unknown";
  }
}

int main(int argc, char** argv) {
  // Parse command line arguments
  auto benchmark_program = argparse::ArgumentParser("benchmark");
  using namespace std::string_view_literals;
  constexpr auto grid_dim_arg = "--grid_dim"sv;
  constexpr auto block_dim_arg = "--block_dim"sv;
  constexpr auto delta_l_arg = "--delta_l"sv;
  constexpr auto time_step_arg = "--time_step"sv;
  constexpr auto type_arg = "--type"sv;
  constexpr auto type_only_updator = 0;
  constexpr auto type_updator_and_tfsf = 1;
  constexpr auto type_updator_and_tfsf_and_pml = 2;
  constexpr auto type_updator_and_tfsf_and_pml_and_nf2ff = 3;
  benchmark_program.add_argument(grid_dim_arg, "-g")
      .help("Grid dimension")
      .nargs(3)
      .default_value(std::vector<int>{4, 4, 4})
      .action([](const std::string& value) { return std::stoi(value); });
  benchmark_program.add_argument(block_dim_arg, "-b")
      .help("Block dimension")
      .nargs(3)
      .default_value(std::vector<int>{8, 8, 8})
      .action([](const std::string& value) { return std::stoi(value); });
  benchmark_program.add_argument(type_arg, "-t")
      .help("Benchmark type")
      .default_value(0)
      .action([](const std::string& value) { return std::stoi(value); });
  benchmark_program.add_argument(delta_l_arg, "-d")
      .help("Delta l")
      .default_value(2.5e-3)
      .action([](const std::string& value) { return std::stod(value); });
  benchmark_program.add_argument(time_step_arg, "-s")
      .help("Time step")
      .default_value(2400)
      .action([](const std::string& value) { return std::stoi(value); });
  try {
    benchmark_program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << benchmark_program;
    return 1;
  }
  auto grid_dim_vec = benchmark_program.get<std::vector<int>>(grid_dim_arg);
  auto block_dim_vec = benchmark_program.get<std::vector<int>>(block_dim_arg);
  // to dim3
  auto func_to_dim3 = [](const std::vector<int>& vec) {
    auto x = static_cast<unsigned int>(vec[0]);
    auto y = static_cast<unsigned int>(vec[1]);
    auto z = static_cast<unsigned int>(vec[2]);
    if (x == 0) {
      x = 1;
    }
    if (y == 0) {
      y = 1;
    }
    if (z == 0) {
      z = 1;
    }
    return dim3{x, y, z};
  };
  auto grid_dim = func_to_dim3(grid_dim_vec);
  auto block_dim = func_to_dim3(block_dim_vec);
  const auto dl = benchmark_program.get<double>(delta_l_arg);
  const auto time_step = benchmark_program.get<int>(time_step_arg);
  const auto type = benchmark_program.get<int>(type_arg);

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

  auto nx = static_cast<xfdtd::Index>(0.35 / dl);
  auto ny = static_cast<xfdtd::Index>(0.35 / dl);
  auto nz = static_cast<xfdtd::Index>(0.35 / dl);

  auto s{xfdtd::Simulation{dl, dl, dl, 0.9, xfdtd::ThreadConfig{1, 1, 1}}};
  s.addObject(domain);
  s.addObject(dielectric_sphere);

  if (3 <= type) {
    constexpr std::size_t nffft_start{static_cast<size_t>(11)};
    auto nffft_fd{std::make_shared<xfdtd::NFFFTFrequencyDomain>(
        nffft_start, nffft_start, nffft_start, xt::xarray<double>{1e9},
        (data_path / "fd").string())};
    s.addNF2FF(nffft_fd);
  }
  if (2 <= type) {
    s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XN));
    s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XP));
    s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YN));
    s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YP));
    s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZN));
    s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZP));
    nx += 16;
    ny += 16;
    nz += 16;
  }
  if (1 <= type) {
    auto l_min{dl * 20};
    auto tau{l_min / 6e8};
    auto t_0{4.5 * tau};
    constexpr std::size_t tfsf_start{static_cast<std::size_t>(15)};
    auto tfsf{std::make_shared<xfdtd::TFSF3D>(
        tfsf_start, tfsf_start, tfsf_start, 0, 0, 0,
        xfdtd::Waveform::gaussian(tau, t_0))};
    s.addWaveformSource(tfsf);
  }

  std::cout << "Grid size: (" << grid_dim.x << ", " << grid_dim.y << ", "
            << grid_dim.z << "), Block size: (" << block_dim.x << ", "
            << block_dim.y << ", " << block_dim.z << ")\n";
  std::cout << "Delta l: " << dl << "\n";
  std::cout << "Time step: " << time_step << "\n";
  std::cout << "Number of grid: " << nx << "x" << ny << "x" << nz << " : "
            << nx * ny * nz << "\n";
  std::cout << "Type: " << typeToString(type) << "\n";

  auto start_time = std::chrono::high_resolution_clock::now();

  auto s_hd = xfdtd::cuda::SimulationHD{&s};
  s_hd.setGridDim(grid_dim);
  s_hd.setBlockDim(block_dim);
  s_hd.run(time_step);

  if (3 <= type) {
    auto nffft_fd = std::dynamic_pointer_cast<xfdtd::NFFFTFrequencyDomain>(
        s.nf2ffs().front());
    if (nffft_fd == nullptr) {
      std::cerr << "Failed to cast NFFFTFrequencyDomain\n";
    } else {
      auto tfsf =
          std::dynamic_pointer_cast<xfdtd::TFSF3D>(s.waveformSources().front());
      if (tfsf == nullptr) {
        std::cerr << "Failed to cast TFSF3D\n";
      } else {
        auto time = tfsf->waveform()->time();
        auto incident_wave_data = tfsf->waveform()->value();
        xt::dump_npy((data_path / "time.npy").string(), time);
        xt::dump_npy((data_path / "incident_wave.npy").string(),
                     incident_wave_data);
      }

      nffft_fd->processFarField(
          xt::linspace<double>(-xfdtd::constant::PI, xfdtd::constant::PI, 360),
          0, "xz");
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> time_span = end_time - start_time;
  std::cout << "Total elapsed time: " << time_span.count() << " ms\n";
  std::cout << "Total elapsed time: " << time_span.count() / 1000 << " s\n";
  std::cout << "Total number of grid: "
            << s.gridSpace()->sizeX() * s.gridSpace()->sizeY() *
                   s.gridSpace()->sizeZ()
            << "\n";
  std::cout << "Per second for grid: "
            << time_step * s.gridSpace()->sizeX() * s.gridSpace()->sizeY() *
                   s.gridSpace()->sizeZ() / (time_span.count() / 1000)
            << "\n";
  return 0;
}
