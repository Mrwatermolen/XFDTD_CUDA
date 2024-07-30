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

#include "xfdtd_cuda/simulation/simulation_hd.cuh"

auto benchmarkOnlyUpdator(dim3 grid_dim, dim3 block_dim) {
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

    benchmarkOnlyUpdator(grid_dim, block_dim);
    return 0;
}
