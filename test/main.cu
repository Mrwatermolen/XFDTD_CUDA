#include <xfdtd/boundary/pml.h>

#include <cstdio>
#include <limits>
#include <memory>
#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>

#include "xfdtd/common/constant.h"
#include "xfdtd/coordinate_system/coordinate_system.h"
#include "xfdtd/electromagnetic_field/electromagnetic_field.h"
#include "xfdtd/monitor/field_monitor.h"
#include "xfdtd/monitor/movie_monitor.h"
#include "xfdtd/object/object.h"
#include "xfdtd/parallel/mpi_support.h"
#include "xfdtd/parallel/parallelized_config.h"
#include "xfdtd/shape/cube.h"
#include "xfdtd/shape/cylinder.h"
#include "xfdtd/simulation/simulation.h"
#include "xfdtd/waveform/waveform.h"
#include "xfdtd/waveform_source/tfsf_2d.h"

void cylinderScatter2D() {
  // constexpr double center_frequency{12e9};
  constexpr double max_frequency{20e9};
  constexpr double min_lambda{3e8 / max_frequency};
  // constexpr double bandwidth{2 * center_frequency};
  constexpr double dx{min_lambda / 20};
  constexpr double dy{dx};
  // constexpr double tau{1.7 / (max_frequency - center_frequency)};
  // constexpr double t_0{0.8 * tau};
  // constexpr double cylinder_radius{0.03};

  auto domain{std::make_shared<xfdtd::Object>(
      "domain",
      std::make_unique<xfdtd::Cube>(
          xfdtd::Vector{-5 * dx, -5 * dy,
                        -std::numeric_limits<double>::infinity()},
          xfdtd::Vector{10 * dx, 10 * dy,
                        std::numeric_limits<double>::infinity()}),
      xfdtd::Material::createAir())};

  // auto cylinder{std::make_shared<xfdtd::Object>(
  //     "cylinder",
  //     std::make_unique<xfdtd::Cylinder>(
  //         xfdtd::Vector{0.0, 0.0, -std::numeric_limits<double>::infinity()},
  //         cylinder_radius, std::numeric_limits<double>::infinity(),
  //         xfdtd::Axis::Axis::ZP),
  //     xfdtd::Material::createPec())};

  // auto tfsf_2d{std::make_shared<xfdtd::TFSF2D>(
  //     50, 50, xfdtd::constant::PI * 0.25,
  //     xfdtd::Waveform::cosineModulatedGaussian(tau, t_0, center_frequency))};

  // auto movie{std::make_shared<xfdtd::MovieMonitor>(
  //     std::make_unique<xfdtd::FieldMonitor>(
  //         std::make_unique<xfdtd::Cube>(
  //             xfdtd::Vector{-175 * dx, -175 * dy,
  //                           -std::numeric_limits<double>::infinity()},
  //             xfdtd::Vector{330 * dx, 350 * dy, xfdtd::constant::INF}),
  //         xfdtd::EMF::Field::EZ, "", ""),
  //     20, "movie", "./data/cylinder_scatter_2d")};
  auto s{xfdtd::Simulation{dx, dy, 1, 0.8, xfdtd::ThreadConfig{1, 1, 1}}};
  s.addObject(domain);
  // s.addObject(cylinder);
  // s.addBoundary(std::make_shared<xfdtd::PML>(10,
  // xfdtd::Axis::Direction::XN));
  // s.addBoundary(std::make_shared<xfdtd::PML>(10,
  // xfdtd::Axis::Direction::XP));
  // s.addBoundary(std::make_shared<xfdtd::PML>(10,
  // xfdtd::Axis::Direction::YN));
  // s.addBoundary(std::make_shared<xfdtd::PML>(10,
  // xfdtd::Axis::Direction::YP)); s.addWaveformSource(tfsf_2d);
  // s.addMonitor(movie);
  s.init(150);

  // {
  //   auto &&t = s.calculationParam()->timeParam().get();

  //   auto t_hd = xfdtd::cuda::TimeParamHD{t};
  //   t_hd.copyHostToDevice();
  //   xfdtd::cuda::__kernelCheckTimeParam<<<1, 1>>>(t_hd.device());
  //   cudaDeviceSynchronize();
  //   t_hd.copyDeviceToHost();
  //   printf("TimeParam::dt=%.5e\n", t->dt());
  //   printf("start_time_step:%lu, current_time_step:%lu\n",
  //   t->startTimeStep(),
  //          t->currentTimeStep());
  // }

  // {
  //   auto &&f = s.calculationParam()->fdtdCoefficient().get();
  //   auto f_hd = xfdtd::cuda::FDTDCoefficientHD{f};
  //   f_hd.copyHostToDevice();
  //   xfdtd::cuda::__kernelCheckFDTDUpdateCoefficient<<<1, 1>>>(f_hd.device());
  //   cudaDeviceSynchronize();
  //   f_hd.copyDeviceToHost();
  // }

  // {
  //   auto c = s.calculationParam().get();
  //   auto c_hd = xfdtd::cuda::CalculationParamHD{c};
  //   c_hd.copyHostToDevice();
  //   xfdtd::cuda::__kernelCheckTimeParam<<<1,
  //   1>>>(c_hd.timeParamHD().device()); cudaDeviceSynchronize();
  //   xfdtd::cuda::__kernelCheckFDTDUpdateCoefficient<<<1, 1>>>(
  //       c_hd.fdtdCoefficientHD().device());
  //   cudaDeviceSynchronize();
  //   c_hd.copyDeviceToHost();
  //   printf("TimeParam::dt=%.5e\n", c->timeParam()->dt());
  //   printf("start_time_step:%lu, current_time_step:%lu\n",
  //          c->timeParam()->startTimeStep(),
  //          c->timeParam()->currentTimeStep());
  // }

  // {
  //   auto g = s.gridSpace().get();
  //   auto g_hd = xfdtd::cuda::GridSpaceHD{g};
  //   g_hd.copyHostToDevice();
  //   xfdtd::cuda::__kenerlCheckGridSpace<<<1, 1>>>(g_hd.device());
  //   cudaDeviceSynchronize();
  // }
}

int main(int argc, char *argv[]) {
  cylinderScatter2D();
  return 0;
}
