#include <xfdtd/boundary/pml.h>
#include <xfdtd/common/constant.h>
#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/material/material.h>
#include <xfdtd/nffft/nffft_frequency_domain.h>
#include <xfdtd/object/object.h>
#include <xfdtd/shape/sphere.h>
#include <xfdtd/simulation/simulation.h>
#include <xfdtd/waveform_source/tfsf_3d.h>

#include <iostream>
#include <memory>
#include <xfdtd_cuda/simulation/simulation_hd.cuh>
#include <xtensor/xnpy.hpp>

#include "argparse/argparse.hpp"

class Visitor : public xfdtd::SimulationFlagVisitor {
 public:
  explicit Visitor(const xfdtd::Simulation& s) : _s{s} {}

  auto initStep(xfdtd::SimulationInitFlag flag) -> void override {
    if (flag == xfdtd::SimulationInitFlag::UpdateStart) {
      _start = std::chrono::high_resolution_clock::now();
    } else if (flag == xfdtd::SimulationInitFlag::UpdateEnd) {
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - _start)
              .count();
      std::cout << "Elapsed time: " << duration << " ms" << std::endl;
      const auto s = &_s;
      if (s == nullptr) {
        return;
      }

      std::cout << "(" << s->gridSpace()->globalGridSpace()->sizeX() << ", "
                << s->gridSpace()->globalGridSpace()->sizeY() << ", "
                << s->gridSpace()->globalGridSpace()->sizeZ() << ")"
                << std::endl;
      auto number_grid = s->gridSpace()->globalGridSpace()->sizeX() *
                         s->gridSpace()->globalGridSpace()->sizeY() *
                         s->gridSpace()->globalGridSpace()->sizeZ();

      std::cout << "The number of grid: " << number_grid << '\n';
      auto e = end - _start;
      auto nano =
          std::chrono::duration_cast<std::chrono::microseconds>(e).count();
      auto time_steps = s->calculationParam()->timeParam()->endTimeStep() -
                        s->calculationParam()->timeParam()->startTimeStep();
      auto per_second_grid_m =
          time_steps * number_grid / (static_cast<double>(nano) / 1e6) / 1e6;

      std::cout << "Grid per second: " << per_second_grid_m << " MCell/s\n";
    }
  }

  auto iteratorStep(xfdtd::SimulationIteratorFlag flag, xfdtd::Index cur,
                    xfdtd::Index start, xfdtd::Index end) -> void override {}

 private:
  std::chrono::high_resolution_clock::time_point _start;
  const xfdtd::Simulation& _s;
};

int main(int argc, char** argv) {
  // Parse command line arguments
  auto benchmark_program = argparse::ArgumentParser("benchmark");
  using namespace std::string_view_literals;
  constexpr auto grid_dim_arg = "--grid_dim"sv;
  constexpr auto block_dim_arg = "--block_dim"sv;
  constexpr auto number_grid_arg = "--num_grid"sv;
  constexpr auto time_step_arg = "--time_step"sv;

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
  benchmark_program.add_argument(number_grid_arg, "-n")
      .help("The number of grid in one direction")
      .default_value(64)
      .scan<'d', int>();
  benchmark_program.add_argument(time_step_arg, "-t")
      .help("Time step")
      .default_value(400)
      .scan<'d', int>();
  try {
    benchmark_program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << '\n';
    std::cerr << benchmark_program;
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
  const xfdtd::Real dl = 5e-3;
  const auto n = benchmark_program.get<int>(number_grid_arg);
  const auto time_step = benchmark_program.get<int>(time_step_arg);

  auto x_size = n * dl;
  auto y_size = x_size;
  auto z_size = x_size;
  auto x_min = -x_size / 2;
  auto y_min = x_min;
  auto z_min = x_min;

  auto domain{std::make_shared<xfdtd::Object>(
      "domain",
      std::make_unique<xfdtd::Cube>(xfdtd::Vector{x_min, y_min, z_min},
                                    xfdtd::Vector{x_size, y_size, z_size}),
      xfdtd::Material::createAir())};
  std::cout << "Time steps: " << time_step << '\n';
  std::cout << "grid Dim: " << grid_dim.x << " " << grid_dim.y << " "
            << grid_dim.z << "\n";
  std::cout << "block Dim: " << block_dim.x << " " << block_dim.y << " "
            << block_dim.z << "\n";

  auto s{xfdtd::Simulation{dl, dl, dl, 0.9, xfdtd::ThreadConfig{1, 1, 1}}};
  s.addObject(domain);
  s.addVisitor(std::make_shared<Visitor>(s));

  auto s_hd = xfdtd::cuda::SimulationHD{&s};
  s_hd.setGridDim(grid_dim);
  s_hd.setBlockDim(block_dim);
  s_hd.run(time_step);

  return 0;
}
