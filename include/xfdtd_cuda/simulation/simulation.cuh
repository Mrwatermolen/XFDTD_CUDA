#ifndef __XFDTD_CUDA_SIMULATION_CUH__
#define __XFDTD_CUDA_SIMULATION_CUH__

#include <cstdio>
#include <memory>
#include <xfdtd/grid_space/grid_space.h>

#include "xfdtd_cuda/common.cuh"
// #include "xfdtd/cuda/grid_space/grid_space.cuh"
#include "xfdtd_cuda/index_task.cuh"
#include "xfdtd_cuda/tensor_hd.cuh"

namespace xfdtd {

namespace cuda {

struct SimulationData {

  XFDTD_CUDA_DUAL auto task() const -> const IndexTask & { return _task; }

  IndexTask _task;
};

XFDTD_CUDA_GLOBAL auto
__kernelSimulationInit(SimulationData *simulation) -> void {
  const auto &x_range = simulation->task().xRange();
  const auto &y_range = simulation->task().yRange();
  const auto &z_range = simulation->task().zRange();

  printf("BlockIdx: %d, %d, %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
  printf("ThreadIdx: %d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
  printf("X Range: %d, %d\n", x_range.start(), x_range.end());
  printf("Y Range: %d, %d\n", y_range.start(), y_range.end());
  printf("Z Range: %d, %d\n", z_range.start(), z_range.end());
}

class Simulation {
public:
  friend class SimulationHD;

  Simulation(std::shared_ptr<xfdtd::GridSpace> grid_space)
      : _grid_space(grid_space) {}

  auto init() -> void;

  auto run() -> void {
    xfdtd::cuda::IndexTask task;
    auto start_x = 0UL;
    auto start_y = 0UL;
    auto start_z = 0UL;
    auto end_x = _grid_space->sizeX();
    auto end_y = _grid_space->sizeY();
    auto end_z = _grid_space->sizeZ();
    std::cout << "Start X: " << start_x << std::endl;
    std::cout << "Start Y: " << start_y << std::endl;
    std::cout << "Start Z: " << start_z << std::endl;
    std::cout << "End X: " << end_x << std::endl;
    std::cout << "End Y: " << end_y << std::endl;
    std::cout << "End Z: " << end_z << std::endl;

    auto s_h = xfdtd::cuda::SimulationData{
        xfdtd::cuda::IndexTask{xfdtd::cuda::IndexRange{start_x, end_x},
                               xfdtd::cuda::IndexRange{start_y, end_y},
                               xfdtd::cuda::IndexRange{start_z, end_z}}};

    std::cout << "Start X: " << s_h.task().xRange().start() << std::endl;
    std::cout << "Start Y: " << s_h.task().yRange().start() << std::endl;
    std::cout << "Start Z: " << s_h.task().zRange().start() << std::endl;
    std::cout << "End X: " << s_h.task().xRange().end() << std::endl;
    std::cout << "End Y: " << s_h.task().yRange().end() << std::endl;
    std::cout << "End Z: " << s_h.task().zRange().end() << std::endl;

    xfdtd::cuda::SimulationData *s_d;

    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaMalloc(&s_d, sizeof(SimulationData)));

    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
        cudaMemcpy(s_d, &s_h, sizeof(SimulationData), cudaMemcpyHostToDevice));
    auto block_size = dim3(1, 1, 1);
    __kernelSimulationInit<<<1, block_size>>>(s_d);
    cudaDeviceSynchronize();
  }

  XFDTD_CUDA_DUAL auto task() const { return _task; }

private:
  std::shared_ptr<xfdtd::GridSpace> _grid_space;

  IndexTask _task;
};

class SimulationHD {
public:
  using Host = xfdtd::cuda::Simulation;

private:
};

} // namespace cuda

} // namespace xfdtd

#endif // __XFDTD_CUDA_SIMULATION_CUH__
