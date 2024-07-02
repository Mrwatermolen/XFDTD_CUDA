#ifndef __XFDTD_CUDA_DOMAIN_CUH__
#define __XFDTD_CUDA_DOMAIN_CUH__

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/grid_space/grid_space.cuh>
#include <xfdtd_cuda/index_task.cuh>

namespace xfdtd::cuda {

class Domain {
 public:
  XFDTD_CUDA_DEVICE auto task() const -> IndexTask { return _task; }

  XFDTD_CUDA_DEVICE auto run() -> void;

  XFDTD_CUDA_DEVICE auto updateH() -> void {};

  XFDTD_CUDA_DEVICE auto correctH() -> void {};

  XFDTD_CUDA_DEVICE auto updateE() -> void {};

  XFDTD_CUDA_DEVICE auto correctE() -> void {};

  XFDTD_CUDA_DEVICE auto record() -> void {};

  XFDTD_CUDA_DEVICE auto nextStep() -> void {};

  IndexTask _task;
  GridSpaceData _grid_space;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DOMAIN_CUH__
