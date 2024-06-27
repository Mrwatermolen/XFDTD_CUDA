#ifndef __XFDTD_CUDA_DOMAIN_CUH__
#define __XFDTD_CUDA_DOMAIN_CUH__

#include "xfdtd/cuda/common.cuh"
#include "xfdtd/cuda/grid_space/grid_space.cuh"
#include "xfdtd/cuda/index_task.cuh"
namespace xfdtd {
namespace cuda {

class Domain {
 public:
  XFDTD_CUDA_DUAL auto task() const -> IndexTask { return _task; }

  XFDTD_CUDA_DUAL auto run() -> void;

  XFDTD_CUDA_DUAL auto updateH() -> void {};

  XFDTD_CUDA_DUAL auto correctH() -> void {};

  XFDTD_CUDA_DUAL auto updateE() -> void {};

  XFDTD_CUDA_DUAL auto correctE() -> void {};

  XFDTD_CUDA_DUAL auto record() -> void {};

  XFDTD_CUDA_DUAL auto nextStep() -> void {};

  IndexTask _task;
  GridSpaceData _grid_space;
};

auto Domain::run() -> void {
#ifdef __CUDACC__
  auto id_x = blockIdx.x * blockDim.x + threadIdx.x;
  auto id_y = blockIdx.y * blockDim.y + threadIdx.y;
  printf("id_x: %d, id_y: %d. Task: x: (%lu, %lu), y: (%lu, %lu)\n", id_x, id_y,
         _task.xRange().start(), _task.xRange().end(), _task.yRange().start(),
         _task.yRange().end());
#else

#endif

  updateH();

  correctH();

  updateE();

  correctE();

  record();

  nextStep();
}

}  // namespace cuda
}  // namespace xfdtd

#endif  // __XFDTD_CUDA_DOMAIN_CUH__
