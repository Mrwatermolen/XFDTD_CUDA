#include <xfdtd/boundary/pml.h>

#include <chrono>
#include <cstdio>
#include <limits>
#include <memory>
#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/calculation_param/time_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>
#include <xfdtd_cuda/simulation/simulation_hd.cuh>

#include "xfdtd/common/constant.h"
#include "xfdtd/coordinate_system/coordinate_system.h"
#include "xfdtd/electromagnetic_field/electromagnetic_field.h"
#include "xfdtd/monitor/field_monitor.h"
#include "xfdtd/monitor/movie_monitor.h"
#include "xfdtd/object/object.h"
#include "xfdtd/parallel/parallelized_config.h"
#include "xfdtd/shape/cube.h"
#include "xfdtd/shape/cylinder.h"
#include "xfdtd/simulation/simulation.h"
#include "xfdtd/waveform/waveform.h"
#include "xfdtd/waveform_source/tfsf_2d.h"

void cylinderScatter2D(dim3 grid_size, dim3 block_size) {
  constexpr xfdtd::Real center_frequency{12e9};
  constexpr xfdtd::Real max_frequency{20e9};
  constexpr xfdtd::Real min_lambda{3e8 / max_frequency};
  // constexpr xfdtd::Real bandwidth{2 * center_frequency};
  constexpr xfdtd::Real dx{min_lambda / 20};
  constexpr xfdtd::Real dy{dx};
  constexpr xfdtd::Real tau{1.7 / (max_frequency - center_frequency)};
  constexpr xfdtd::Real t_0{0.8 * tau};
  constexpr xfdtd::Real cylinder_radius{0.03};

  auto domain{std::make_shared<xfdtd::Object>(
      "domain",
      std::make_unique<xfdtd::Cube>(
          xfdtd::Vector{-175 * dx, -175 * dy,
                        -std::numeric_limits<xfdtd::Real>::infinity()},
          xfdtd::Vector{325 * dx, 325 * dy,
                        std::numeric_limits<xfdtd::Real>::infinity()}),
      xfdtd::Material::createAir())};

  // auto domain{std::make_shared<xfdtd::Object>(
  //     "domain",
  //     std::make_unique<xfdtd::Cube>(
  //         xfdtd::Vector{-10 * dx, -10 * dy,
  //                       -std::numeric_limits<xfdtd::Real>::infinity()},
  //         xfdtd::Vector{20 * dx, 20 * dy,
  //                       std::numeric_limits<xfdtd::Real>::infinity()}),
  //     xfdtd::Material::createAir())};

  auto cylinder{std::make_shared<xfdtd::Object>(
      "cylinder",
      std::make_unique<xfdtd::Cylinder>(
          xfdtd::Vector{0.0, 0.0, -std::numeric_limits<xfdtd::Real>::infinity()},
          cylinder_radius, std::numeric_limits<xfdtd::Real>::infinity(),
          xfdtd::Axis::Axis::ZP),
      xfdtd::Material::createPec())};

  auto tfsf_2d{std::make_shared<xfdtd::TFSF2D>(
      50, 50, xfdtd::constant::PI * 0.25,
      xfdtd::Waveform::cosineModulatedGaussian(tau, t_0, center_frequency))};

  auto movie{std::make_shared<xfdtd::MovieMonitor>(
      std::make_unique<xfdtd::FieldMonitor>(
          std::make_unique<xfdtd::Cube>(
              xfdtd::Vector{-175 * dx, -175 * dy,
                            -std::numeric_limits<xfdtd::Real>::infinity()},
              xfdtd::Vector{325 * dx, 325 * dy, xfdtd::constant::INF}),
          xfdtd::EMF::Field::EZ, "", ""),
      40, "movie", "./tmp/cylinder_scatter_2d")};

  // auto movie{std::make_shared<xfdtd::MovieMonitor>(
  //     std::make_unique<xfdtd::FieldMonitor>(
  //         std::make_unique<xfdtd::Cube>(
  //             xfdtd::Vector{-10 * dx, -10 * dy,
  //                           -std::numeric_limits<xfdtd::Real>::infinity()},
  //             xfdtd::Vector{20 * dx, 20 * dy, xfdtd::constant::INF}),
  //         xfdtd::EMF::Field::EZ, "", ""),
  //     20, "movie", "./tmp/cylinder_scatter_2d")};
  auto s{xfdtd::Simulation{dx, dy, 1, 0.8, xfdtd::ThreadConfig{1, 1, 1}}};
  //   s.addObject(domain);
  //   s.addObject(cylinder);
  // s.addBoundary(std::make_shared<xfdtd::PML>(10,
  // xfdtd::Axis::Direction::XN));
  // s.addBoundary(std::make_shared<xfdtd::PML>(10,
  // xfdtd::Axis::Direction::XP));
  // s.addBoundary(std::make_shared<xfdtd::PML>(10,
  // xfdtd::Axis::Direction::YN));
  // s.addBoundary(std::make_shared<xfdtd::PML>(10,
  // xfdtd::Axis::Direction::YP)); s.addWaveformSource(tfsf_2d);
  // s.addMonitor(movie);
  // s.init(15);

  //   {
  //     s.init(15);
  //     auto &&t = s.calculationParam()->timeParam().get();

  //     auto t_hd = xfdtd::cuda::TimeParamHD{t};
  //     t_hd.copyHostToDevice();
  //     xfdtd::cuda::__kernelCheckTimeParam<<<1, 1>>>(t_hd.device());
  //     cudaDeviceSynchronize();
  //     t_hd.copyDeviceToHost();
  //     printf("TimeParam::dt=%.5e\n", t->dt());
  //     printf("start_time_step:%lu, current_time_step:%lu\n",
  //     t->startTimeStep(),
  //            t->currentTimeStep());
  //     return;
  //   }

  // {
  //   s.init(15);
  //   auto &&f = s.calculationParam()->fdtdCoefficient().get();
  //   auto f_hd = xfdtd::cuda::FDTDCoefficientHD{f};
  //   f_hd.copyHostToDevice();
  //   xfdtd::cuda::__kernelCheckFDTDUpdateCoefficient<<<1, 1>>>(f_hd.device());
  //   cudaDeviceSynchronize();
  //   f_hd.copyDeviceToHost();
  // }

  //   {
  //     s.init(15);
  //     auto c = s.calculationParam().get();
  //     auto c_hd = xfdtd::cuda::CalculationParamHD{c};
  //     c_hd.copyHostToDevice();
  //     xfdtd::cuda::__kernelCheckTimeParam<<<1,
  //     1>>>(c_hd.timeParamHD()->device()); cudaDeviceSynchronize();
  //     xfdtd::cuda::__kernelCheckFDTDUpdateCoefficient<<<1, 1>>>(
  //         c_hd.fdtdCoefficientHD()->device());
  //     cudaDeviceSynchronize();
  //     c_hd.copyDeviceToHost();
  //     printf("TimeParam::dt=%.5e\n", c->timeParam()->dt());
  //     printf("start_time_step:%lu, current_time_step:%lu\n",
  //            c->timeParam()->startTimeStep(),
  //            c->timeParam()->currentTimeStep());
  //   }

  //   {
  //     s.init(15);
  //     auto g = s.gridSpace().get();
  //     auto g_hd = xfdtd::cuda::GridSpaceHD{g};
  //     g_hd.copyHostToDevice();
  //     xfdtd::cuda::__kenerlCheckGridSpace<<<1, 1>>>(g_hd.device());
  //     cudaDeviceSynchronize();
  //   }

  {
    std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();
    auto s_hd = xfdtd::cuda::SimulationHD{&s};
    s_hd.host()->addObject(domain);
    s_hd.host()->addObject(cylinder);
    s_hd.host()->addWaveformSource(tfsf_2d);
    s_hd.host()->addBoundary(
        std::make_shared<xfdtd::PML>(10, xfdtd::Axis::Direction::XN));
    s_hd.host()->addBoundary(
        std::make_shared<xfdtd::PML>(10, xfdtd::Axis::Direction::XP));
    s_hd.host()->addBoundary(
        std::make_shared<xfdtd::PML>(10, xfdtd::Axis::Direction::YN));
    s_hd.host()->addBoundary(
        std::make_shared<xfdtd::PML>(10, xfdtd::Axis::Direction::YP));
    s_hd.host()->addMonitor(movie);
    s_hd.setGridDim(grid_size);
    s_hd.setBlockDim(block_size);
    s_hd.run(1500);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    std::chrono::high_resolution_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();
    // to ms
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    printf("duration: %ld ms\n", duration);
  }
}

int main(int argc, char *argv[]) {
  unsigned int block_size_x = 16;
  unsigned int block_size_y = 16;
  unsigned int block_size_z = 1;
  unsigned int grid_size_x = 1;
  unsigned int grid_size_y = 1;
  unsigned int grid_size_z = 1;
  if (7 <= argc) {
    block_size_x = atoi(argv[1]);
    block_size_y = atoi(argv[2]);
    block_size_z = atoi(argv[3]);
    grid_size_x = atoi(argv[4]);
    grid_size_y = atoi(argv[5]);
    grid_size_z = atoi(argv[6]);
  }
  auto block_size = dim3{block_size_x, block_size_y, block_size_z};
  auto grid_size = dim3{grid_size_x, grid_size_y, grid_size_z};
  std::printf("Grid size: (%d, %d, %d), Block size: (%d, %d, %d)\n",
              grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y,
              block_size.z);
  cylinderScatter2D(grid_size, block_size);
  return 0;
}
